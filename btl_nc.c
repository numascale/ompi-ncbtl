/*
 * Copyright (c) 2004-2011 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2007 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2006-2007 Voltaire. All rights reserved.
 * Copyright (c) 2009-2012 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2010-2013 Los Alamos National Security, LLC.  
 *                         All rights reserved. 
 * Copyright (c) 2010-2012 IBM Corporation.  All rights reserved.
 * Copyright (c) 2012      Oracle and/or its affiliates.  All rights reserved.
 * Copyright (c) 2013      Intel, Inc. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include <sys/types.h>
#include <sys/stat.h>
#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif  /* HAVE_FCNTL_H */
#include <errno.h>
#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif  /* HAVE_SYS_MMAN_H */

#include <sys/syscall.h>
#include <semaphore.h>

#include "opal/sys/cma.h"
#include "opal/sys/atomic.h"
#include "opal/class/opal_bitmap.h"
#include "opal/util/output.h"
#include "opal/util/printf.h"
#include "opal/mca/hwloc/base/base.h"
#include "orte/util/proc_info.h"
#include "opal/datatype/opal_convertor.h"
#include "ompi/class/ompi_free_list.h"
#include "ompi/mca/btl/btl.h"
#include "ompi/mca/mpool/base/base.h"

#include "btl_nc.h"
#include "ompi/proc/proc.h"

#include "numa.h"
#include "numaif.h"


ssize_t xprocess_vm_writev(pid_t pid, 
                                const struct iovec  *lvec, 
                                unsigned long liovcnt,
                                const struct iovec *rvec,
                                unsigned long riovcnt,
                                unsigned long flags)
{
  return syscall(310, pid, lvec, liovcnt, rvec, riovcnt, flags);
}

static void* helper_thread(void* arg);
static inline void mfence();
static inline void nccopy4(void* to, const uint32_t head);
static inline void nccopy(void* to, const void* from, size_t n);
static struct mca_btl_base_endpoint_t* create_nc_endpoint(int local_proc, struct ompi_proc_t *proc);
static int nc_btl_first_time_init(mca_btl_nc_t* nc_btl, int n);
static int sendring(int peer, void* data, uint32_t size, uint32_t type, uint32_t seqno);
static int createstat(int n);
static void setstatistics(const uint32_t size);
static void print_stat();
static void sbitset(void* sbits, void* data, int size8, int sbit);
static uint8_t* allocring(int peer_numanode, int size8, int* sbit);
static int send_msg(int peer_numanode, frag_t* frag);
static void fifo_push_back(fifolist_t* list, frag_t* frag);
static void fifo_push_back_n(fifolist_t* list, frag_t* frag);

frag_t* allocfrag(int size);
void freefrag(frag_t* frag);


void INThandler(int sig);

#define NC_SUCCESS 1
#define NC_FAILED  0


static void bind_cpu(int id) {

    cpu_set_t cpuset;

    CPU_ZERO(&cpuset);
    CPU_SET(id, &cpuset);

    pthread_t thread = pthread_self();

    int rc = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if( rc != 0 ) {
        fprintf(stderr, "pthread_setaffinity_np rc = %d", rc);
    }
}


static void bind_node(int node) 
{
    struct bitmask* nodemask;

    nodemask = numa_allocate_nodemask();
    numa_bitmask_clearall(nodemask);
    numa_bitmask_setbit(nodemask, node);
    numa_bind(nodemask);
    numa_bitmask_free(nodemask);
}


void INThandler(int sig) 
{
	signal(sig, SIG_IGN);
	
	print_stat();

	signal(SIGUSR1, INThandler);
}


mca_btl_nc_t mca_btl_nc = {
    {
        &mca_btl_nc_component.super,
        0, /* btl_eager_limit */
        0, /* btl_rndv_eager_limit */
        0, /* btl_max_send_size */
        0, /* btl_rdma_pipeline_send_length */
        0, /* btl_rdma_pipeline_frag_size */
        0, /* btl_min_rdma_pipeline_size */
        0, /* btl_exclusivity */
        0, /* btl_latency */
        0, /* btl_bandwidth */
        0, /* btl flags */
        0, /* btl segment size */
        mca_btl_nc_add_procs,
        mca_btl_nc_del_procs,
        NULL,
        mca_btl_nc_finalize,
        mca_btl_nc_alloc,
        mca_btl_nc_free,
        mca_btl_nc_prepare_src,
        NULL,
        mca_btl_nc_send,
        mca_btl_nc_sendi,
        NULL,  /* put -- optionally filled during initialization */
        NULL,  /* get */
        mca_btl_base_dump,
        NULL, /* mpool */
        mca_btl_nc_register_error_cb, /* register error */
        mca_btl_nc_ft_event
    }
};


int mca_btl_nc_ft_event(int state) 
{
    return OMPI_SUCCESS;
}


static uint64_t inline rdtscp(uint32_t* id) 
{
    uint32_t lo, hi, aux;
	__asm__ __volatile__ (".byte 0x0f,0x01,0xf9" : "=a" (lo), "=d" (hi), "=c" (aux));
	*id = aux;
    return (uint64_t)hi << 32 | lo;
}


static int currCPU()
{
	uint32_t cpuid; 
	rdtscp(&cpuid);
	cpuid &= 0xfff; 
	return cpuid;
}


static int currNumaNode()
{
	uint32_t cpuid; 
	rdtscp(&cpuid);
	return (cpuid >> 12);
}


int getnode(void* ptr) 
{
    void* pageaddrs[1];
    int pagenode;
    int err;

    pagenode = -1;
    pageaddrs[0] = ptr;

    err = syscall(__NR_move_pages, 0, 1, pageaddrs, NULL, &pagenode, 0);
    if (err < 0) {
        perror("__NR_move_pages");
    }

    if (pagenode == -ENOENT) {
        printf("  page 0x%lx is not allocated\n", (uint64_t)ptr);
    }

    return pagenode;
}


static void* map_shm(const int peer, const int shmsize)
{
    char* nc_ctl_file;

	if( peer >= 0 ) {
		if( asprintf(&nc_ctl_file, "%s"OPAL_PATH_SEP"nc_btl_module.%s.%d",
					 orte_process_info.job_session_dir,
					 orte_process_info.nodename,
					 peer) < 0 ) {
			return 0;
		}
	}
	else {
		if( asprintf(&nc_ctl_file, "%s"OPAL_PATH_SEP"nc_btl_module.%s.sys",
					 orte_process_info.job_session_dir,
					 orte_process_info.nodename) < 0 ) {
			return 0;
		}
	}

	int flags = O_RDWR;
	struct stat sbuf;
    int ret = stat(nc_ctl_file, &sbuf);
    if( ret < 0 && errno == ENOENT ) {
        flags |= O_CREAT;
    }

    int shared_fd = open(nc_ctl_file, flags, 0660);
    if( shared_fd < 0 ) {
        printf("open shared file failed. Aborting.\n");
        return 0;
    }

    if( 0 != ftruncate(shared_fd, shmsize) ) {
        printf("failed to set the size of a shared file. Aborting.\n");
        return 0;
    }

    void* shm = mmap(NULL, shmsize, (PROT_READ | PROT_WRITE), MAP_SHARED, shared_fd, 0);

    if( MAP_FAILED == shm ) {
        fprintf(stderr, "failed to mmap a shared file %s. Aborting.\n", nc_ctl_file);
        return 0;
    }

    return shm;
}


int mca_btl_nc_add_procs(
    struct mca_btl_base_module_t* btl,
    size_t nprocs,
    struct ompi_proc_t **procs,
    struct mca_btl_base_endpoint_t **peers,
    opal_bitmap_t* reachability)
{
    int return_code = OMPI_SUCCESS;
    int32_t n_local_procs = 0, proc, j, my_smp_rank = -1;
    ompi_proc_t* my_proc; /* pointer to caller's proc structure */
    mca_btl_nc_t *nc_btl;
    bool have_connected_peer = false;
    char **bases;

    /* initializion */

    nc_btl = (mca_btl_nc_t *)btl;

    /* get pointer to my proc structure */
    if(NULL == (my_proc = ompi_proc_local()))
        return OMPI_ERR_OUT_OF_RESOURCE;

    /* Get unique host identifier for each process in the list,
     * and idetify procs that are on this host.  Add procs on this
     * host to shared memory reachbility list.  Also, get number
     * of local procs in the procs list. */
    for (proc = 0; proc < (int32_t)nprocs; proc++) {
        /* check to see if this proc can be reached via shmem (i.e.,
           if they're on my local host and in my job) */
        if (procs[proc]->proc_name.jobid != my_proc->proc_name.jobid ||
            !OPAL_PROC_ON_LOCAL_NODE(procs[proc]->proc_flags)) {
            peers[proc] = NULL;
            continue;
        }

        /* check to see if this is me */
        if(my_proc == procs[proc]) {
            my_smp_rank = MY_RANK = n_local_procs++;
            continue;
        }

         /* sm doesn't support heterogeneous yet... */
        if (procs[proc]->proc_arch != my_proc->proc_arch) {
            continue;
        }

        /* we have someone to talk to */
        have_connected_peer = true;

        if(!(peers[proc] = create_nc_endpoint(n_local_procs, procs[proc]))) {
            return_code = OMPI_ERROR;
            goto CLEANUP;
        }
        n_local_procs++;

        /* add this proc to shared memory accessibility list */
        return_code = opal_bitmap_set_bit(reachability, proc);
        if(OMPI_SUCCESS != return_code)
            goto CLEANUP;
    }

    /* jump out if there's not someone we can talk to */
    if (!have_connected_peer)
        goto CLEANUP;

    /* make sure that my_smp_rank has been defined */
    if (-1 == my_smp_rank) {
        return_code = OMPI_ERROR;
        goto CLEANUP;
    }

    if (!nc_btl->btl_inited) {
        return_code =
            nc_btl_first_time_init(nc_btl, n_local_procs);
        if (return_code != OMPI_SUCCESS) {
            goto CLEANUP;
        }
    }

    /* set local proc's smp rank in the peers structure for
     * rapid access and calculate reachability */
    for(proc = 0; proc < (int32_t)nprocs; proc++) {
        if(NULL == peers[proc])
            continue;
        peers[proc]->my_smp_rank = my_smp_rank;
    }

    opal_atomic_wmb();

    /* update the local smp process count */
    mca_btl_nc_component.num_smp_procs += n_local_procs;

    if (OMPI_SUCCESS != return_code)
        goto CLEANUP;

CLEANUP:
    return return_code;
}


static int nc_btl_first_time_init(mca_btl_nc_t* nc_btl, int n)
{
	assert( sizeof(frag_t) <= FRAG_SIZE );

    size_t pagesize = sysconf(_SC_PAGESIZE);
	int max_nodes = numa_num_configured_nodes();

	size_t stailsize = ((max_nodes * sizeof(NC_CLSIZE) + pagesize - 1) & ~(pagesize - 1));
	size_t noderecsize = ((sizeof(node_t) + pagesize - 1) & ~(pagesize - 1));

	size_t syssize = ((sizeof(sysctxt_t) + pagesize - 1) & ~(pagesize - 1));
	size_t shmsize = syssize + (max_nodes * stailsize) + (max_nodes * noderecsize);

	// -------------------------------------------------------------------
	sysctxt_t* sysctxt = (sysctxt_t*)map_shm(-1, shmsize);
	if( sysctxt == NULL ) {
		return OMPI_ERR_OUT_OF_RESOURCE;
	}
	mca_btl_nc_component.sysctxt = sysctxt;

	sysctxt->max_nodes = max_nodes;

	int numanode = currNumaNode();

	mca_btl_nc_component.numanode = numanode;
	mca_btl_nc_component.cpuid = currCPU();
	sysctxt->cpuid[MY_RANK] = mca_btl_nc_component.cpuid + 1;

	uint8_t* node0 = (uint8_t*)sysctxt + syssize + (max_nodes * stailsize);
	node_t* mynode = (node_t*)(node0 + numanode * noderecsize);

	int32_t cpuindex = lockedAdd(&(mynode->ndxmax), 1) - 1;

	mca_btl_nc_component.map = sysctxt->map;
	mca_btl_nc_component.map[MY_RANK] = ((((uint32_t)numanode) << 16) | cpuindex);
	int rank_count = lockedAdd(&(sysctxt->rank_count), 1);

	mca_btl_nc_component.cpuindex = cpuindex;	

	mca_btl_nc_component.peer_node = (node_t**)malloc(max_nodes * sizeof(void*));
	for( int i = 0; i < max_nodes; i++ ) {
		mca_btl_nc_component.peer_node[i] = (node_t*)(node0 + i * noderecsize);
	}

	mca_btl_nc_component.node = mca_btl_nc_component.peer_node[numanode];
	volatile node_t* node = mca_btl_nc_component.node;

	// init pointers to send tails in shared mem
	// these pointers will be used by the receiver to reset the send tail on the sender side
	mca_btl_nc_component.peer_stail = (uint32_t**)malloc(max_nodes * sizeof(void*));
	for( int i = 0; i < max_nodes; i++ ) {
		mca_btl_nc_component.peer_stail[i] = (uint32_t*)((uint8_t*)sysctxt + syssize + (i * stailsize));
	}

	mca_btl_nc_component.stail = (uint32_t**)malloc(max_nodes * sizeof(void*));
	uint32_t* stail = mca_btl_nc_component.peer_stail[numanode];
	for( int i = 0; i < max_nodes; i++ ) {
		// use one cache line per counter
		mca_btl_nc_component.stail[i] = stail + i * (NC_CLSIZE >> 2);
	}

	/*
	ring descriptors
	peer ring descriptors
	pending sends lists
	input lists
	rings
	small fragments
	big fragments
	*/

	shmsize = max_nodes * sizeof(ring_t);
	shmsize += max_nodes * sizeof(pring_t);
	shmsize += max_nodes * sizeof(fifolist_t);
	shmsize += max_nodes * sizeof(fifolist_t);
	shmsize = (shmsize + pagesize - 1) & ~(pagesize - 1);

	size_t ring_ofs = shmsize;

	shmsize += (size_t)max_nodes * (size_t)RING_SIZE;

	size_t frag_ofs = shmsize;

	shmsize += MAX_SIZE_FRAGS; 
	shmsize = (shmsize + pagesize - 1) & ~(pagesize - 1);

	// map local mem
	void* shmbase = map_shm(numanode, shmsize);
	assert( shmbase );
	mca_btl_nc_component.shmsize = shmsize;
	mca_btl_nc_component.shm_base = shmbase;
	mca_btl_nc_component.ring_ofs = ring_ofs;
	mca_btl_nc_component.frag_ofs = frag_ofs;
	mca_btl_nc_component.shm_ringbase = shmbase + ring_ofs;
	mca_btl_nc_component.shm_fragbase = shmbase + frag_ofs;

	// init list of pointers to local peer ring descriptors
	mca_btl_nc_component.ring_desc = (ring_t*)shmbase;

	// init list of pointers to local peer ring descriptors
	mca_btl_nc_component.peer_ring_desc = (pring_t*)(shmbase + max_nodes * sizeof(ring_t));

	mca_btl_nc_component.peer_ring_buf = (void**)malloc(max_nodes * sizeof(void*));
	memset(mca_btl_nc_component.peer_ring_buf, 0, max_nodes * sizeof(void*));

	mca_btl_nc_component.pending_sends = (fifolist_t*)(
		shmbase + max_nodes * sizeof(ring_t) + max_nodes * sizeof(pring_t));

	// local input queue
	mca_btl_nc_component.inq = shmbase 
								+ max_nodes * sizeof(ring_t)
								+ max_nodes * sizeof(pring_t)
								+ max_nodes * sizeof(fifolist_t);

	mca_btl_nc_component.myinq = mca_btl_nc_component.inq + cpuindex;

	if( cpuindex == 0 ) {	
		pthread_create(&mca_btl_nc_component.sendthread, 0, &helper_thread, 0);
	}

	// wait until helper thread ready
	while( !node->active );

	// offset for local processes to local fragments
	mca_btl_nc_component.shm_ofs = shmbase - node->shm_base;
	mca_btl_nc_component.frag0 = allocfrag(BFRAG_SIZE - sizeof(frag_t));
	assert( mca_btl_nc_component.frag0 );

	fprintf(stderr, "RANK %d, TOATL %d, PID %d, NODE %d, CPUID %d, CPU INDEX %d, shmsize %d MB, shm_base %p, shm_ofs %p\n", 
		MY_RANK,
		sysctxt->rank_count,
		getpid(),
		mca_btl_nc_component.numanode,
		mca_btl_nc_component.cpuid,
		mca_btl_nc_component.cpuindex,
		(int)shmsize / 1024 / 1024,
		mca_btl_nc_component.shm_base,
		(void*)mca_btl_nc_component.shm_ofs);
	fflush(stderr);

	// -------------------------------------------------------------------

    char* ev = getenv("NCSTAT");
    mca_btl_nc_component.statistics = (ev && !strcasecmp(ev, "yes"));
    if( mca_btl_nc_component.statistics ) {
        mca_btl_nc_component.statistics = createstat(n);
    }

	if( MY_RANK == 0 ) {
		printf("************USING NC-BTL 1.8.3************\n");
	}

	signal(SIGUSR1, INThandler);

//    assert( lzo_init() == LZO_E_OK );

    /* set flag indicating btl has been inited */
    nc_btl->btl_inited = true;
	mfence();

    return OMPI_SUCCESS;
}


struct mca_btl_base_endpoint_t*
create_nc_endpoint(int local_proc, struct ompi_proc_t *proc)
{
    struct mca_btl_base_endpoint_t *ep;

    ep = (struct mca_btl_base_endpoint_t*)
        malloc(sizeof(struct mca_btl_base_endpoint_t));
    if(NULL == ep)
        return NULL;
    ep->peer_smp_rank = local_proc + mca_btl_nc_component.num_smp_procs;
    return ep;
}


static int createstat(int n) 
{
    int flags = O_RDWR;
    char* filename;
    size_t pagesize = sysconf(_SC_PAGESIZE);
    size_t size = n * pagesize;
    struct stat sbuf;

    if( asprintf(&filename, "%s"OPAL_PATH_SEP"nc_btl_module_stat.%s",
                 orte_process_info.job_session_dir,
                 orte_process_info.nodename) < 0 ) {
        return 0;
    }

    int ret = stat(filename, &sbuf);
    if( ret < 0 && errno == ENOENT ) {
        flags |= O_CREAT;
    }

    int shared_fd = open(filename, flags, 0600);
    if( shared_fd < 0 ) {
        fprintf(stderr, "open %s failed...\n", filename);
        return 0;
    }

    if( 0 != ftruncate(shared_fd, size) ) {
        fprintf(stderr, "failed to set the size of a shared file. Aborting.\n");
        return 0;
    }

    uint8_t* shm = mmap(NULL, size, (PROT_READ | PROT_WRITE), MAP_SHARED, shared_fd, 0);

    if( MAP_FAILED == shm ) {
        fprintf(stderr, "failed to mmap a shared file %s\n", filename);
        return 0;
    }

    mca_btl_nc_component.shm_stat = shm + MY_RANK * pagesize;

    memset(mca_btl_nc_component.shm_stat, 0, pagesize);

    struct msgstats_t* pstat = (struct msgstats_t*)mca_btl_nc_component.shm_stat;
    pstat->maxrank = n - 1;
    pstat->rank = MY_RANK;
    pstat->bytessend = 0;
    pstat->totmsgs = 0;
    pstat->active = 1;
    return 1;
}


int mca_btl_nc_del_procs(
    struct mca_btl_base_module_t* btl,
    size_t nprocs,
    struct ompi_proc_t **procs,
    struct mca_btl_base_endpoint_t **peers)
{
    return OMPI_SUCCESS;
}


/**
 * MCA->BTL Clean up any resources held by BTL module
 * before the module is unloaded.
 *
 * @param btl (IN)   BTL module.
 *
 * Prior to unloading a BTL module, the MCA framework will call
 * the BTL finalize method of the module. Any resources held by
 * the BTL should be released and if required the memory corresponding
 * to the BTL module freed.
 *
 */

int mca_btl_nc_finalize(struct mca_btl_base_module_t* btl)
{
    return OMPI_SUCCESS;
}


/*
 * Register callback function for error handling..
 */
int mca_btl_nc_register_error_cb(
        struct mca_btl_base_module_t* btl,
        mca_btl_base_module_error_cb_fn_t cbfunc)
{
    mca_btl_nc_t* nc_btl = (mca_btl_nc_t*)btl;
    nc_btl->error_cb = cbfunc;
    return OMPI_SUCCESS;
}


static void setstatistics(const uint32_t size)
{
    struct msgstats_t* pstat = (struct msgstats_t*)mca_btl_nc_component.shm_stat;

    assert( size > 0 );

    uint32_t i;

    // find bit position of msb in size
    __asm__ __volatile__ (
        "bsr %1, %%ebx\n"
        "movl %%ebx, %0\n"
        : "=r" (i) : "r" (size) : "ebx");

	if( size & (size - 1) ) {
		++i;
	}

    ++pstat->dist[i];
    ++pstat->totmsgs;
    pstat->bytessend += size;
}


static void print_stat()
{
	if( mca_btl_nc_component.statistics && (MY_RANK == 0) ) {

		int i, j, k;

        size_t pagesize = sysconf(_SC_PAGESIZE);

        int m = mca_btl_nc_component.num_smp_procs; // rows

        // find largest message send
        int maxndx = 0;
        for( i = 0; i < m; i++ ) {
            struct msgstats_t* pstat = (struct msgstats_t*)(mca_btl_nc_component.shm_stat + i * pagesize);
            for( j = 4; j < 32; j++ ) {
                if( pstat->dist[j] > 0 ) {
                    if( j > maxndx ) maxndx = j;
                }
            }
        }

        int n = 3 + (maxndx + 1 - 4); // cols,start at with 16 == index 4

        char** tab = (char**)malloc(m * n * sizeof(char*));
		uint64_t totmsgs = 0;
		uint64_t totsend = 0;

        // create strings for all size values in pstat->dist
        char val[256];
        for( i = 0; i < m; i++ ) {

            struct msgstats_t* pstat = (struct msgstats_t*)(mca_btl_nc_component.shm_stat + i * pagesize);

			totmsgs += pstat->totmsgs;
			totsend += pstat->bytessend;

            for( j = 0; j < n; j++ ) {
                if( j == 0 ) {
                    sprintf(val, "%d", i);
                }
                else
                if( j == 1 ) {
                    sprintf(val, "%ld", (int64_t)pstat->totmsgs);
                }
                else
                if( j == 2 ) {
                    sprintf(val, "%ld", (int64_t)pstat->bytessend / 1024 / 1000);
                }
                else {
                    sprintf(val, "%ld", (int64_t)pstat->dist[j - 3 + 4]); // start with size 16 == index 4
                }
                char* str = malloc(strlen(val) + 1);
                strcpy(str, val);
                tab[i * n + j] = str;
            }
        }

        static const char* hdr[32] = {
            "rank", "send msgs", "send MB",
            "<=16", "<=32", "<=64", "<=128", "<=256", "<=512", "<=1K", "<=2K", "<=4K",
            "<=8K", "<=16K", "<=32K", "<=64K", "<=128K", "<=256K", "<=512K", "<=1M", "<=2M",
            "<=4M", "<=8M", "<=16M", "<=32M", "<=64M", "<=128M", "<=256M", "<=512M",
            "<=1G", "<=2G", "<=4G"};

        // find max width of all strings per column
        for( j = 0; j < n; j++ ) {
            int w = strlen(hdr[j]);
            for( i = 0; i < m; i++ ) {
                char* str = tab[i * n + j];
                if( strlen(str) > w ) {
                    w = strlen(str);
                }
            }
            ++w;

            // prepend strings with blancs so there are all same width per column
            for( i = 0; i < m; i++ ) {
                char* str = tab[i * n + j];
                char* str2 = malloc(w + 1);
                memset(str2, ' ', w);
                strcpy(str2 + w - strlen(str), str);
                free(str);
                tab[i * n + j] = str2;
            }
        }

        char line[4096];
        line[0] = 0;
        k = 0;
        for( j = 0; j < n; j++ ) {
            int w = strlen(tab[j]);
            memset(line + k, ' ', w);
            strcpy(line + k + w - strlen(hdr[j]), hdr[j]);
            k += w;
        }
        fprintf(stderr, "\n\n%s\n", line);
        memset(line, '-', k);
        line[k] = 0;
        fprintf(stderr, "%s\n", line);

        for( i = 0; i < m; i++ ) {
            line[0] = 0;
            for( j = 0; j < n; j++ ) {
                char* str = tab[i * n + j];
                strcat(line, str);
                free(str);
            }
            fprintf(stderr, "%s\n", line);
            fflush(stderr);
        }

        memset(line, '-', k);
        line[k] = 0;
        fprintf(stderr, "%s\n", line);
		
        fprintf(stderr, "%ld msgs, %ld MB\n", (int64_t)totmsgs, (int64_t)(totsend / 1024 / 1000));

        fprintf(stderr, "\n\n");
        fflush(stderr);
        free(tab);
 	}
}


/**
 * Allocate a segment.
 *
 * @param btl (IN)      BTL module
 * @param size (IN)     Request segment size.
 */
extern mca_btl_base_descriptor_t* mca_btl_nc_alloc(
    struct mca_btl_base_module_t* btl,
    struct mca_btl_base_endpoint_t* endpoint,
    uint8_t order,
    size_t size,
    uint32_t flags)
{
	uint32_t size8 = ((sizeof(mca_btl_nc_hdr_t) + size + 7) & ~7);
	int ssize = syncsize(size8);
	uint32_t rsize = sizeof(rhdr_t) + ssize + size8;

	frag_t* frag = allocfrag(rsize);
	if( !frag ) {
		return 0;
	}
		
	frag->hdr_size = size;
    mca_btl_nc_hdr_t* hdr = (mca_btl_nc_hdr_t*)((uint8_t*)(frag + 1) + sizeof(rhdr_t) + ssize);

	hdr->frag = frag;
    hdr->base.des_src = &hdr->segment.base;
    hdr->base.des_src_cnt = 1;
    hdr->base.des_dst = &hdr->segment.base;
    hdr->base.des_dst_cnt = 1;
    hdr->segment.base.seg_len = size;
    hdr->base.des_flags = flags;
    hdr->segment.base.seg_addr.pval = hdr + 1;

	frag->hdr_size = size8;
	frag->size = sizeof(mca_btl_nc_hdr_t) + size;
	frag->rsize = rsize;

    return (mca_btl_base_descriptor_t*)hdr;
}

/**
 * Return a segment allocated by this BTL.
 *
 * @param btl (IN)      BTL module
 * @param segment (IN)  Allocated segment.
 */
extern int mca_btl_nc_free(
    struct mca_btl_base_module_t* btl,
    mca_btl_base_descriptor_t* des)
{
    return OMPI_SUCCESS;
}


/**
 * Initiate an inline send to the peer. If failure then return a descriptor.
 *
 * @param btl (IN)      BTL module
 * @param peer (IN)     BTL peer addressing
 */
int mca_btl_nc_sendi( struct mca_btl_base_module_t* btl,
                      struct mca_btl_base_endpoint_t* endpoint,
                      struct opal_convertor_t* convertor,
                      void* header,
                      size_t header_size,
                      size_t payload_size,
                      uint8_t order,
                      uint32_t flags,
                      mca_btl_base_tag_t tag,
                      mca_btl_base_descriptor_t** descriptor )
{
	int peer = endpoint->peer_smp_rank;
    int size = payload_size + header_size;

	int size8 = ((size + 7) & ~7);
	int ssize = syncsize(size8);
	int rsize = sizeof(rhdr_t) + ssize + size8;

//fprintf(stderr, "%d -> %d ISEND, size %d %d %d\n", MY_RANK, peer, (int)payload_size, (int)header_size, rsize);
//fflush(stderr);

	frag_t* frag = allocfrag(rsize);
	if( !frag ) {
		// upper layers will call progress() first and than try sending again
		return OMPI_ERR_RESOURCE_BUSY;
	}

	rhdr_t* rhdr = (rhdr_t*)(frag + 1);

	uint32_t dst = mca_btl_nc_component.map[peer];
	int peer_numanode = (dst >> 16);

	// if there are already pending sends to peer, this message must be queued also
	bool queued = (bool)mca_btl_nc_component.pending_sends[peer_numanode].head;

	int sbit;
	uint8_t* rbuf;

	if( !queued ) {
		rbuf = allocring(peer_numanode, rsize, &sbit);
		if( !rbuf ) {
			queued = true;
		}
	}

    rhdr->type = MSG_TYPE_ISEND;
	rhdr->dst_ndx = (dst & 0xffff); 
	rhdr->rsize = (rsize >> 2);
	rhdr->pad8 = (size8 - size);

	assert( rhdr->rsize > 0 );

	uint8_t* data = (uint8_t*)(rhdr + 1) + ssize;

	if( header_size ) {
		memcpy(data, header, header_size);
	}

	if( payload_size ) {
		size_t max_data;
		uint32_t iov_count = 1;
		struct iovec iov;
		iov.iov_len = max_data = payload_size;
		iov.iov_base = data + header_size;

		int rc = opal_convertor_pack(convertor, &iov, &iov_count, &max_data);
		if( rc < 0 ) {
			freefrag(frag);
			// upper layers will call progress() first and than try sending again
			return OMPI_ERR_RESOURCE_BUSY;
		}
	}

	if( queued ) {
		frag->size = size;
		frag->rsize = rsize;
		fifolist_t* list = &(mca_btl_nc_component.pending_sends[peer_numanode]);
		fifo_push_back(list, frag);
		return OMPI_SUCCESS;
	}

    rhdr->type |= sbit;

	if( size8 ) {
		void* sbits = (size8 <= SHDR) ? (void*)&rhdr->sbits : (void*)(rhdr + 1);
		sbitset(sbits, data, size8, sbit);
	}

	nccopy(rbuf, rhdr, rsize);

	freefrag(frag);

	*descriptor = 0;

    if( mca_btl_nc_component.statistics ) {
        setstatistics(size);
    }
	return OMPI_SUCCESS;
}


/**
 * Pack data
 *
 * @param btl (IN)      BTL module
 */
struct mca_btl_base_descriptor_t* mca_btl_nc_prepare_src(
    struct mca_btl_base_module_t* btl,
    struct mca_btl_base_endpoint_t* endpoint,
    mca_mpool_base_registration_t* registration,
    struct opal_convertor_t* convertor,
    uint8_t order,
    size_t reserve,
    size_t* size,
    uint32_t flags)
{
    size_t max_data = *size;
	node_t* node = mca_btl_nc_component.node;

	// reserve is used for specific header e.g. MATCH or RDMA headers
	int hdr_size = sizeof(mca_btl_nc_hdr_t) + reserve;
	int msgsize = hdr_size + max_data;

	int sz = msgsize;

	bool cont = (msgsize > MAX_SEND_SIZE);
	if( cont ) {
		sz = MAX_SEND_SIZE;
	}

	int size8 = ((sz + 7) & ~7);
	int ssize = syncsize(size8);
	int rsize = sizeof(rhdr_t) + ssize + size8;

//fprintf(stderr, "%d PREPARE, ssize %d, rsize %d, msgsize %d, sz %d\n", 
//	MY_RANK, ssize, rsize, msgsize, sz);
//fflush(stderr);

	frag_t* frag0 = allocfrag(rsize);
	frag0->sbit = 5;
	assert( frag0 );

	mca_btl_nc_hdr_t* hdr = (mca_btl_nc_hdr_t*)((uint8_t*)(frag0 + 1) + sizeof(rhdr_t) + ssize);
	frag0->hdr_size = sizeof(mca_btl_nc_hdr_t) + reserve;

	struct iovec iov;
	uint32_t iov_count = 1;
	iov.iov_len = sz - hdr_size;

	iov.iov_base = (uint8_t*)hdr + hdr_size;
	frag_t* frag = frag0;

	int rest = max_data;
	while( rest ) {
		frag->next = 0;
		frag->size = sz;
		frag->rsize = rsize;

		size_t sz1;
		int rc = opal_convertor_pack(convertor, &iov, &iov_count, &sz1);

		assert( rc >= 0 );
		assert( iov.iov_len == sz1 );
		assert( sz1 > 0 );

		rest -= (int)sz1;

//fprintf(stderr, "%d, REST %d\n", MY_RANK, rest);
//fflush(stderr);

		if( rest <= 0 ) {
			assert( rest == 0 );
			break;
		}

		if( rest < sz ) {
			sz = rest;
			size8 = ((sz + 7) & ~7);
			ssize = syncsize(size8);
			rsize = sizeof(rhdr_t) + ssize + size8;
		}

		frag_t* next;
		
		next = allocfrag(rsize);
		assert( next );

		frag->next = next;
		frag = next;
		iov.iov_base = (uint8_t*)(frag + 1) + sizeof(rhdr_t) + ssize;
		iov.iov_len = sz;
	}

	hdr->frag = frag0;

	hdr->segment.base.seg_addr.pval = hdr + 1; 
    hdr->segment.base.seg_len = reserve + max_data;

    hdr->base.des_src = &(hdr->segment.base);
    hdr->base.des_src_cnt = 1;
    hdr->base.order = MCA_BTL_NO_ORDER;
    hdr->base.des_dst = NULL;
    hdr->base.des_dst_cnt = 0;
    hdr->base.des_flags = flags;

	*size = max_data;

    return &(hdr->base);
}


/**
 * Initiate a send to the peer.
 *
 * @param btl (IN)      BTL module
 * @param peer (IN)     BTL peer addressing
 */
int mca_btl_nc_send( struct mca_btl_base_module_t* btl,
                     struct mca_btl_base_endpoint_t* endpoint,
                     struct mca_btl_base_descriptor_t* descriptor,
                     mca_btl_base_tag_t tag )
{
	mca_btl_nc_hdr_t* hdr = (mca_btl_nc_hdr_t*)descriptor;
    hdr->tag = tag;
   	hdr->endpoint = endpoint;
	hdr->src_rank = MY_RANK;

	frag_t* frag0 = hdr->frag; 
	frag_t* frag = frag0;

//fprintf(stderr, "%d SEND, n = %d, frag %p, frag->id %d, frag->size %d...\n", MY_RANK, frag->d1, frag, frag->id, frag->size);
//fflush(stderr);

	if( MCA_BTL_DES_SEND_ALWAYS_CALLBACK & hdr->base.des_flags ) {
		// create ack hdr (contains a copy of header for use in ack message)
		int hdr_size = frag->hdr_size;
		mca_btl_nc_hdr_t* ackhdr = (mca_btl_nc_hdr_t*)malloc(hdr_size);
		if( !ackhdr ) {
			return NC_FAILED;				
		}

		hdr->self = ackhdr;
		memcpy(ackhdr, hdr, hdr_size);

		ackhdr->segment.base.seg_addr.pval = ackhdr + 1; 
		ackhdr->base.des_src = &(ackhdr->segment.base);
	}

	int peer = endpoint->peer_smp_rank;
	assert( peer >= 0 && peer < mca_btl_nc_component.sysctxt->rank_count );
	uint32_t dst = mca_btl_nc_component.map[peer];

	bool cont = (bool)frag0->next;
	int type = cont ? MSG_TYPE_BLK0 : MSG_TYPE_FRAG;

	for( ; ; ) {
		rhdr_t* rhdr = (rhdr_t*)(frag + 1);
		rhdr->type = type;
		rhdr->dst_ndx = (dst & 0xffff);

		frag = frag->next;
		if( !frag ) {
			break;
		}

		if( frag->next ) {
			type = MSG_TYPE_BLK;
		}
		else {
			type = MSG_TYPE_BLKN;
		}
	}

	int peer_numanode = (dst >> 16);
	fifolist_t* list = &(mca_btl_nc_component.pending_sends[peer_numanode]);

	if( cont ) {
		fifo_push_back_n(list, frag0);
	}
	else {
		fifo_push_back(list, frag0);
	}
	return NC_SUCCESS;
}


/**
 *
 */
void mca_btl_nc_dump(struct mca_btl_base_module_t* btl,
                     struct mca_btl_base_endpoint_t* endpoint,
                     int verbose)
{
}


static int send_msg(int peer_numanode, frag_t* frag)
{
	rhdr_t* rhdr = (rhdr_t*)(frag + 1);

	uint32_t size = frag->size;
	uint32_t size8 = ((size + 7) & ~7);
	uint32_t ssize = syncsize(size8); 
	uint32_t rsize = frag->rsize;

	assert( sizeof(rhdr) + ssize + size8 == rsize );

	int sbit;
	uint8_t* rbuf = allocring(peer_numanode, rsize, &sbit);
	if( !rbuf ) {
		return NC_FAILED;
	}

	rhdr->type |= sbit;
	rhdr->rsize = (rsize >> 2);
	rhdr->pad8 = (size8 - size);
	assert( size8 - size <= 7 );

	assert( rhdr->rsize > 0 );

	if( size8 ) {
		void* sbits = (size8 <= SHDR) ? (void*)&rhdr->sbits : (void*)(rhdr + 1);
		void* data = (uint8_t*)(rhdr + 1) + ssize;
		sbitset(sbits, data, size8, sbit);
	}

	nccopy(rbuf, rhdr, rsize);

	return NC_SUCCESS;
}


void sendack(int peer, void* hdr)
{
	uint32_t dst = mca_btl_nc_component.map[peer];
	int peer_numanode = (dst >> 16);
	uint32_t rsize = sizeof(rhdr_t) + sizeof(void*);

	fifolist_t* list = &(mca_btl_nc_component.pending_sends[peer_numanode]);

	// if there are already pending sends to peer, this message must be queued also
	bool queued = false; //(bool)list->head;

	int sbit;
	uint8_t* rbuf;

	if( !queued ) {
		rbuf = allocring(peer_numanode, rsize, &sbit);
		if( !rbuf ) {
			queued = true;
		}
	}

	if( queued ) {
		frag_t* frag = allocfrag(rsize);

		rhdr_t* rhdr = (rhdr_t*)(frag + 1);

		rhdr->type = MSG_TYPE_ACK;
		rhdr->dst_ndx = (dst & 0xffff);
		rhdr->sbits = 0;

		uint8_t* data = (uint8_t*)(rhdr + 1);
		*(void**)data = hdr;

		frag->size = sizeof(void*);
		frag->rsize = rsize;

		fifo_push_back(list, frag);
		return;
	}

	uint8_t buf[rsize] __attribute__((aligned(8)));

	rhdr_t* rhdr = (rhdr_t*)buf;

    rhdr->type = (MSG_TYPE_ACK | sbit);
	rhdr->dst_ndx = (dst & 0xffff);
	rhdr->rsize = (rsize >> 2);
	rhdr->sbits = 0;
	rhdr->pad8 = 0;

	assert( rhdr->rsize > 0 );

	uint8_t* data = (uint8_t*)(rhdr + 1);
	*(void**)data = hdr;

	if( sbit ) {
		*((uint64_t*)data) |= 1;
	}

	nccopy(rbuf, buf, rsize);
}


static void nccopy4(void* to, const uint32_t head)
{
    __asm__ __volatile__ (
		"movl %0, %%eax\n"
		"movnti %%eax, (%1)\n"
		"sfence\n"
        :: "r" (head), "r" (to) : "eax", "memory");
}


static void nccopy(void* to, const void* from, size_t n)
{
    assert( n > 0 );
	assert( (n & 0x7) == 0 );
	assert( (((uint64_t)from) & 0x7) == 0 );
	assert( (((uint64_t)to) & 0x7) == 0 );

	n >>= 3;

	for( size_t i = 0; i < n; i++ ) {
		__asm__ __volatile__ (
			"movq (%0), %%rax\n"
			"movnti %%rax, (%1)\n"
		    : : "r" (from), "r" (to) : "rax", "memory");
		from += 8;
		to += 8;
	}

    __asm__ __volatile__ ("sfence");
}


frag_t* allocfrag(int size)
{
	assert( size <= BFRAG_SIZE - sizeof(frag_t) );
	bool small = (size <= SFRAG_SIZE - sizeof(frag_t));

	node_t* node = mca_btl_nc_component.node;
	fragpool_t* pool = &node->fragpool;
	fifolist_t* freelist = small ? &pool->sfrags : &pool->bfrags;

	frag_t* frag = 0;
	uint64_t ofs = mca_btl_nc_component.shm_ofs;
	__semlock(&freelist->lock);

	if( freelist->head ) {
		frag = (frag_t*)((void*)freelist->head + ofs);
		freelist->head = frag->next;
		__semunlock(&freelist->lock);
	}
	else {
		__semunlock(&freelist->lock);

		int size = small ? SFRAG_SIZE : BFRAG_SIZE;

		__semlock(&pool->lock);

		// allocate frag in shared mem
		if( size <= pool->frags_rest ) {
			frag = (frag_t*)(pool->shm_frags_low + ofs);
			frag->small = small;
			frag->inuse = false; 
			pool->shm_frags_low += size;
			pool->frags_rest -= size;
		}
		__semunlock(&pool->lock);
	}	

assert( frag );

if( frag ) {
	frag->next = 0;
}

	return frag;
}


void freefrag(frag_t* frag)
{
	fragpool_t* pool = &(mca_btl_nc_component.node->fragpool);
	fifolist_t* list = (frag->small) ? &pool->sfrags : &pool->bfrags;

	// transfer fragment to local nodes address space
	frag_t* newhead = (frag_t*)((void*)frag - mca_btl_nc_component.shm_ofs);

	__semlock(&list->lock);

    assert( newhead != list->head );
	frag->next = list->head;
	list->head = newhead;

	__semunlock(&list->lock);
}


// append to fifo list, multiple producers, single consumer
static void fifo_push_back(fifolist_t* list, frag_t* frag)
{
	frag->next = 0;
	uint64_t ofs = mca_btl_nc_component.shm_ofs;

	// transfer fragment to local nodes address space
	frag = (frag_t*)((void*)frag - ofs);

	__semlock(&list->lock);

	if( list->head ) {
		frag_t* tail = (frag_t*)((void*)list->tail + ofs);
		tail->next = frag;
	}
	else {
		list->head = frag;
	}
	list->tail = frag;

	__semunlock(&list->lock);
}


// append to fifo list, multiple producers, single consumer
static void fifo_push_back_n(fifolist_t* list, frag_t* frag)
{
	uint64_t ofs = mca_btl_nc_component.shm_ofs;

	// transfer fragments to local nodes address space
	frag_t* tail = frag;
	frag_t* next = tail->next;

	while( next ) {
		tail->next = (frag_t*)((void*)next - ofs);
		tail = next;
		next = tail->next;
	}
	assert( tail );
	assert( tail != frag );
	assert( tail->next == 0 );

	frag = (frag_t*)((void*)frag - ofs);
	tail = (frag_t*)((void*)tail - ofs);

	__semlock(&list->lock);

	if( list->head ) {
		frag_t* t = (frag_t*)((void*)list->tail + ofs);
		t->next = frag;
	}
	else {
		list->head = frag;
	}
	list->tail = tail;
	__semunlock(&list->lock);
}


 static void commit_ring(int rndx)
{
	node_t* node = mca_btl_nc_component.node;

	ring_t* ring = mca_btl_nc_component.ring_desc + node->ring_cnt;

	ring->buf = mca_btl_nc_component.shm_ringbase + (rndx * RING_SIZE);

	assert( ring->buf );
	assert( (((uint64_t)ring->buf) & 0xfff) == 0 );

	memset(ring->buf, 0, RING_SIZE);
	sfence();

	ring->ptail = (uint8_t*)mca_btl_nc_component.peer_stail[rndx]
		+ (mca_btl_nc_component.numanode * 64) - (uint64_t)mca_btl_nc_component.sysctxt; // use one cache line per counter
	sfence();

	++node->ring_cnt;
	sfence();
}


static void init_ring(int peer_numanode)
{
	sysctxt_t* sysctxt = mca_btl_nc_component.sysctxt;
	int loc_numanode = mca_btl_nc_component.numanode;

	void* ring_base;

	if( peer_numanode == loc_numanode ) {
		// send thread on local node has already mapped ring
		ring_base = mca_btl_nc_component.shm_ringbase;
	}
	else {
		ring_base = map_shm(peer_numanode, mca_btl_nc_component.shmsize) + mca_btl_nc_component.ring_ofs;
	}

	volatile pring_t* pr = mca_btl_nc_component.peer_ring_desc + peer_numanode;

	__semlock(&pr->lock);

	if( !pr->commited ) {
		pr->commited = true;

		if( (peer_numanode == loc_numanode) && (mca_btl_nc_component.sendthread == pthread_self()) ) {
			commit_ring(loc_numanode);
		}
		else {
			// pr->buf is pointer to remote ring, address is based on mapping for local nodes process
			node_t* peer_node = mca_btl_nc_component.peer_node[peer_numanode];

			// send request to commit ring on target node
			for( ; ; ) {
				if( __sync_val_compare_and_swap(&peer_node->commit_ring, 0, loc_numanode + 1) == 0 ) {
					break;
				}
			}

			// wait until commit done
			while( peer_node->commit_ring );
		}

		pr->commited = true;
	}

	mca_btl_nc_component.peer_ring_buf[peer_numanode] = ring_base + (loc_numanode * RING_SIZE);

	__semunlock(&pr->lock);
}


static void checkring(int peer_numanode, int size8)
{
	static const int RING_GUARD = 8;

	// peer ring descriptor is in shared mem
	pring_t* pr = mca_btl_nc_component.peer_ring_desc + peer_numanode;

	__semlock(&pr->lock);

	uint32_t head = pr->head;
	uint32_t tail = *(mca_btl_nc_component.stail[peer_numanode]);

	int space = 0;

	if( head >= tail ) {
		if( head + size8 + RING_GUARD <= RING_SIZE ) {
			space = RING_SIZE - head - RING_GUARD;
		}
		else 
		if( size8 + RING_GUARD <= tail ) {
			space = tail - RING_GUARD;
		}
	}
	else {
		if( size8 + RING_GUARD <= (tail - head) ) {
			space = head - tail - RING_GUARD;
		}
	}

	fprintf(stderr, "%d HEAD %d, TAIL %d, SPACE %d\n", MY_RANK, head, tail, space); 
	fflush(stderr);

	__semunlock(&pr->lock);
}


static uint8_t* allocring(int peer_numanode, int size8, int* sbit)
{
	assert( (size8 & 7) == 0 );

	sysctxt_t* sysctxt = mca_btl_nc_component.sysctxt;
	int loc_numanode = mca_btl_nc_component.numanode;

	void* ring_buf = mca_btl_nc_component.peer_ring_buf[peer_numanode];

	if( !ring_buf ) {
		// map target nodes rings
		init_ring(peer_numanode);
		ring_buf = mca_btl_nc_component.peer_ring_buf[peer_numanode];
	}

	static const int RING_GUARD = 8;
	uint8_t* buf = 0;

	// peer ring descriptor is in shared mem
	pring_t* pr = mca_btl_nc_component.peer_ring_desc + peer_numanode;

	__semlock(&pr->lock);

	uint32_t head = pr->head;

	uint32_t tail = *(mca_btl_nc_component.stail[peer_numanode]);

	if( head >= tail ) {
		if( head + size8 + RING_GUARD <= RING_SIZE ) {
			pr->head += size8;
			buf = ring_buf + head;
			*sbit = (pr->sbit ^ 1);
		}
		else 
		if( size8 + RING_GUARD <= tail ) {
			assert( head + RING_GUARD <= RING_SIZE );

			// reset ring

			// toggle syncbit
			*sbit = pr->sbit;
			pr->sbit ^= 1;

			nccopy4(ring_buf + head, MSG_TYPE_RST | pr->sbit);
			pr->head = size8;
			buf = ring_buf;
		}
	}
	else {
		if( size8 + RING_GUARD <= (tail - head) ) {
			pr->head += size8;
			buf = ring_buf + head;
			*sbit = (pr->sbit ^ 1);
		}
	}

	__semunlock(&pr->lock);

	return buf;
}


static void sbitset(void* sbits, void* data, int size8, int sbit)
{
	// pointer to workload
	uint64_t* p = (uint64_t*)data; 
	assert( ((uint64_t)p & 0x7) == 0 );

	int n = (size8 >> 3);

	if( n <= 32 ) {
		uint32_t b = 0;

		for( int i = 0; i < n; i++ ) {
			b <<= 1;
			b |= (uint32_t)((*p) & 1);

			if( sbit ) {
				(*p) |= 1;
			}
			else {
				(*p) &= ~1;
			}
			++p;

		}
		*(uint32_t*)sbits = b;
	}
	else {
		// pointer to sync bits
		uint64_t* q = (uint64_t*)sbits;				 
		assert( ((uint64_t)q & 0x7) == 0 );

		uint64_t b = 0;
		int k = 0;

		for( int i = 0; i < n; i++ ) {
			b |= ((*p) & 1);
			b <<= 1;

			if( sbit ) {
				(*p) |= 1;
			}
			else {
				(*p) &= ~1;
			}
			++p;

			if( ++k >= 63 ) {
				*q++ = (b | sbit);
				b = 0;
				k = 0;
			}
		}
		if( k ) {
			*q = (b | sbit);
		}
	}
}


static void place_helper()
{
	sysctxt_t* sysctxt = mca_btl_nc_component.sysctxt;
	int n = sysctxt->rank_count;
	int numanode = mca_btl_nc_component.numanode;
	int cpu0 = currCPU();

fprintf(stderr, "%d >>>>>>>>>>>>>>>>>>>>>>>>>>>>> place_helper() n %d, numanode %d, cpu cnt %d...\n", MY_RANK, n, numanode, numa_num_configured_cpus());
fflush(stderr);

	assert( numa_node_of_cpu(cpu0) == numanode ); 

	int cpuid = -1;
	for( int id = 0; id < numa_num_configured_cpus(); id++ ) {

		if( numa_node_of_cpu(id) != numanode ) {
			continue;
		}

		bool found = false;
		for( int j = 0; j < n; j++ ) {
			if( sysctxt->cpuid[j] - 1 == id ) {
				found = true;
				break;
			}
		}

		if( !found ) {
			cpuid = id;
			break;
		}
	}

fprintf(stderr, "%d >>>>>>>>>>>>>>>>>>>>>>>>>>>>> cpu0 %d, cpuid %d...\n", MY_RANK, cpu0, cpuid);
fflush(stderr);
	assert( numa_node_of_cpu(cpu0) == numa_node_of_cpu(cpuid) ); 

	if( cpuid >= 0 ) {
		bind_cpu(cpuid); 
	}
}


static void* helper_thread(void* arg)
{
	node_t* node = mca_btl_nc_component.node;

	int locpeercnt = node->ndxmax;
	place_helper();

	// init node structure, done clear last structure member cpuindex
	memset(node, 0, offsetof(node_t, ndxmax));

	memset(mca_btl_nc_component.shm_base, 0, mca_btl_nc_component.ring_ofs);

	sysctxt_t* sysctxt = mca_btl_nc_component.sysctxt;
	int max_nodes = sysctxt->max_nodes;

	node->shm_base = mca_btl_nc_component.shm_base;

	node->fragpool.shm_frags = node->shm_base + mca_btl_nc_component.frag_ofs;
	node->fragpool.shm_frags_low = node->fragpool.shm_frags;
	node->fragpool.frags_rest = MAX_SIZE_FRAGS;

	node->active = 1;
	sfence();

	fprintf(stderr, "%2d (%2d %2d) send_thread()...\n", 
		MY_RANK, currCPU(),	currNumaNode());
	fflush(stderr);

	// pending send lists
	int list_cnt = max_nodes;
	//fifolist_t* lists = (fifolist_t*)malloc(max_nodes * sizeof(void*));
	//memset(lists, 0, max_nodes * sizeof(void*));

	while( node->active ) {

		if( locpeercnt != node->ndxmax ) {
			locpeercnt = node->ndxmax;
			place_helper();
		}

		volatile fifolist_t* list = mca_btl_nc_component.pending_sends;

		bool idle = true;
		// send all pending sends
		for( int n = 0; n < list_cnt; n++ ) {

			while( list->head ) {

				frag_t* frag = list->head;

				if( send_msg(n, frag) ) {
					if( frag->next ) {
						list->head = frag->next;
					}
					else {
						__semlock(&list->lock);
						list->head = frag->next;
						__semunlock(&list->lock);
					}

					freefrag(frag);
				}
				idle = false;
			}
			++list;
		}

		if( node->commit_ring ) {
			// ring commit request
			commit_ring(node->commit_ring - 1);
			node->commit_ring = 0;
		}

		//if( idle ) {
//			sched_yield();
		//}
	}

	return NULL;
}
