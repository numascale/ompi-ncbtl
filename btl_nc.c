/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
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

#define MPOL_BIND 2             // mbind memory binding policy


static void* send_thread(void* arg);
static inline void nccopy(void* to, const void* from, int n);
static struct mca_btl_base_endpoint_t* create_nc_endpoint(int local_proc, struct ompi_proc_t *proc);
static int nc_btl_first_time_init(mca_btl_nc_t* nc_btl, int n);
static int sendring(int peer, void* data, uint32_t size, uint32_t type, uint32_t seqno);
static int createstat(int n);
static void setstatistics(const uint32_t size);
static void print_stat();
static rhdr_t* allocring(int peer_node, int size, int* sbit);
static bool isend_msg(int node, rhdr_t* hdr, void* buf, int size);
static bool send_msg(frag_t* frag);
static void push_sendq_sync(int peer, frag_t* frag);
static void push_sendq_async(int peer, frag_t* frag);
static void scopy(void* dst, const void* src, int size, int sbit);
static void ringbind();


frag_t* allocfrag(int size);
void freefrag(frag_t* frag);
void INThandler(int sig);


static uint64_t inline rdtscp(uint32_t* id)
{
    uint32_t lo, hi, aux;
    __asm__ __volatile__ (".byte 0x0f,0x01,0xf9" : "=a" (lo), "=d" (hi), "=c" (aux));
    *id = aux;
    return (uint64_t)hi << 32 | lo;
}


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


static void* map_shm(bool sys, size_t shmsize)
{
    char* nc_ctl_file;

    if( sys ) {
        if( asprintf(&nc_ctl_file, "%s"OPAL_PATH_SEP"nc_btl_module.%s.sys",
                     orte_process_info.job_session_dir,
                     orte_process_info.nodename) < 0 ) {
            return 0;
        }
    }
    else {
        if( asprintf(&nc_ctl_file, "%s"OPAL_PATH_SEP"nc_btl_module.%s",
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
        fprintf(stderr, "open shared file failed. Aborting.\n");
        fflush(stderr);
        return 0;
    }

    if( 0 != ftruncate(shared_fd, shmsize) ) {
        fprintf(stderr, "failed to set the size of a shared file. Aborting.\n");
        fflush(stderr);
        return 0;
    }

    void* shm = mmap(NULL, shmsize, (PROT_READ | PROT_WRITE), MAP_SHARED, shared_fd, 0);

    if( !shm || (MAP_FAILED == shm) ) {
        fprintf(stderr, "failed to mmap a shared file %s. Aborting.\n", nc_ctl_file);
        fflush(stderr);
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

CLEANUP:
    return return_code;
}


static int nc_btl_first_time_init(mca_btl_nc_t* nc_btl, int n)
{
    /*
    ring descriptors
    peer ring descriptors
    rings
    fragments
    */

    size_t pagesize = sysconf(_SC_PAGESIZE);

    size_t shmsize = n * sizeof(ring_t);
	shmsize += n * sizeof(pring_t);
    shmsize = (shmsize + pagesize - 1) & ~(pagesize - 1);

    size_t ring_ofs = shmsize;
	mca_btl_nc_component.ring_ofs = ring_ofs;

    shmsize += (size_t)n * (size_t)RING_SIZE;
    size_t frag_ofs = shmsize;

    shmsize += MAX_SIZE_FRAGS;
    shmsize = (shmsize + pagesize - 1) & ~(pagesize - 1);

    // map local mem
    void* shm = map_shm(false, n * shmsize);
    assert( shm );

    mca_btl_nc_component.shm_base = (void**)malloc(n * sizeof(void*));
    for( int i = 0; i < n; i++ ) {
		mca_btl_nc_component.shm_base[i] = shm + i * shmsize;
	}

    // init pointers to send tails in shared mem
    // these pointers will be used by the receiver to reset the send tail on the sender side
    mca_btl_nc_component.peer_stail = (uint32_t**)malloc(n * sizeof(void*));
    for( int i = 0; i < n; i++ ) {
		pring_t* pr = (pring_t*)(mca_btl_nc_component.shm_base[i] + (n * sizeof(ring_t)));
        mca_btl_nc_component.peer_stail[i] = &((pr + MY_RANK)->stail);
    }

    void* shmbase = mca_btl_nc_component.shm_base[MY_RANK];

    int rc = madvise(shmbase + frag_ofs + (1024 * 1024),
                     shmsize - frag_ofs - (1024 * 1024),
                     MADV_DONTNEED);
    if( rc != 0 ) {
        fprintf(stderr, "madvise() failed, errno = %d\n", errno);
        fflush(stderr);
    }

    memset(shmbase, 0, n * sizeof(ring_t) + n * sizeof(pring_t));

    mca_btl_nc_component.shm_ringbase = shmbase + ring_ofs;
    mca_btl_nc_component.shm_fragbase = shmbase + frag_ofs;

    // init list of pointers to local peer ring descriptors
    mca_btl_nc_component.peer_ring = (pring_t*)(shmbase + n * sizeof(ring_t));

    mca_btl_nc_component.peer_ring_buf = (void**)malloc(n * sizeof(void*));
    memset(mca_btl_nc_component.peer_ring_buf, 0, n * sizeof(void*));

	ring_t* ring = (ring_t*)shmbase;
    for( int i = 0; i < n; i++ ) {
		ring->peer = -1;
		++ring;
	}

    // local input queue
    mca_btl_nc_component.inq = (fifolist_t*)malloc(n * sizeof(fifolist_t));
	memset(mca_btl_nc_component.inq, 0, n * sizeof(fifolist_t));
    mca_btl_nc_component.myinq = mca_btl_nc_component.inq;

    mca_btl_nc_component.sendqcnt = (int32_t*)malloc(n * sizeof(int32_t));
	memset(mca_btl_nc_component.sendqcnt, 0, n * sizeof(int32_t));

    ringbind();

    // init frag heap
    frag_t* frag = (frag_t*)mca_btl_nc_component.shm_fragbase;
    frag->inuse = false;
    frag->prevsize = 0;
    frag->fsize = MAX_SIZE_FRAGS;
    frag->lastfrag = true;

    if( mca_btl_nc_component.async_send ) {
        pthread_mutex_t* sendq_mutex = &mca_btl_nc_component.send_mutex;
        pthread_mutexattr_t mutex_attr;
        pthread_mutexattr_init(&mutex_attr);
        pthread_mutex_init(sendq_mutex, &mutex_attr);
        pthread_create(&mca_btl_nc_component.sendthread, 0, &send_thread, 0);
    }
    else {
        mca_btl_nc_component.send_thread_active = 1;
        __sfence();
    }

    char* ev = getenv("NCSTAT");
    mca_btl_nc_component.statistics = (ev && (!strcasecmp(ev, "yes") || !strcasecmp(ev, "true") || !strcasecmp(ev, "1")));
    if( mca_btl_nc_component.statistics ) {
        mca_btl_nc_component.statistics = createstat(n);
    }

    while( !mca_btl_nc_component.send_thread_active );

    if( MY_RANK == 0 ) {
        fprintf(stderr, "************ NC-BTL 1.8.x ************\n");
        fprintf(stderr, "-USING ASYNCRONOUS SEND : %s\n", mca_btl_nc_component.async_send ? "YES" : "NO");
        fprintf(stderr, "\n");
        fprintf(stderr, "to modify this use mpiexec parameters :\n");
        fprintf(stderr, "--mca btl_nc_async_send    (0|1)\n");
        fflush(stderr);
    }

    signal(SIGUSR1, INThandler);

    /* set flag indicating btl has been inited */
    __sfence();
    nc_btl->btl_inited = true;

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
	char* name = "/dev/shm/NCMPI_STAT";

	struct stat sbuf;
    //if( stat(name, &sbuf) == 0 ) {
    //    unlink(name);
    //}

	int flags = (O_RDWR | O_CREAT);
    int fd = open(name, flags, 0660);
    if( fd < 0 ) {
		fprintf(stderr, "opening mapping file %s failed\n", name);
		fflush(stderr);
		exit(-1);
    }

	size_t pagesize = sysconf(_SC_PAGESIZE);
    size_t size = n * pagesize;
    if( 0 != ftruncate(fd, size) ) {
 	    fprintf(stderr, "mapping shared file %s failed\n", name);
		fflush(stderr);
		exit(-1);
    }

    void* shm = mmap(NULL, size, (PROT_READ | PROT_WRITE), MAP_SHARED, fd, 0);

    if( MAP_FAILED == shm ) {
 	    fprintf(stderr, "mapping shared file %s failed\n", name);
		fflush(stderr);
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
    mca_btl_nc_component.send_thread_active = 0;
    if( mca_btl_nc_component.async_send ) {
        pthread_cond_signal(&mca_btl_nc_component.send_cond);
    }

    print_stat();
    
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

	__sync_add_and_fetch((uint64_t*)&(pstat->dist[i]), 1);
	__sync_add_and_fetch((uint64_t*)&(pstat->totmsgs), 1);
	__sync_add_and_fetch((uint64_t*)&(pstat->bytessend), size);
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

		if( totmsgs > 0 ) {
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
    int peer = endpoint->peer_smp_rank;
    uint32_t size8 = ((sizeof(mca_btl_nc_hdr_t) + size + 7) & ~7);

    frag_t* frag = allocfrag(RHDR_SIZE + size8);
    mca_btl_nc_hdr_t* hdr = (mca_btl_nc_hdr_t*)(frag + 1);
    hdr = (mca_btl_nc_hdr_t*)((void*)hdr + RHDR_SIZE);

    hdr->frag = frag;
    hdr->base.des_src = &hdr->segment.base;
    hdr->base.des_src_cnt = 1;
    hdr->base.des_dst = &hdr->segment.base;
    hdr->base.des_dst_cnt = 1;
    hdr->segment.base.seg_len = size;
    hdr->base.des_flags = flags;
    hdr->segment.base.seg_addr.pval = hdr + 1;

    frag->size = sizeof(mca_btl_nc_hdr_t) + size;
    frag->send = 0;
    frag->msgtype = MSG_TYPE_FRAG;

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
	int rc = OMPI_SUCCESS;

    int peer = endpoint->peer_smp_rank;
    int size = header_size + payload_size;
    int size8 = ((size + 7) & ~7);

    frag_t* frag = 0;
    assert( size <= MAX_EAGER_SIZE );
    uint8_t ALIGN8 txbuf[MAX_EAGER_SIZE];
    void* data = txbuf;

    // if there are already pending sends to peer, this message must be queued also
    bool queued = mca_btl_nc_component.sendqcnt[peer];

    if( queued ) {
        frag = allocfrag(RHDR_SIZE + size8);
        data = (void*)(frag + 1) + RHDR_SIZE;
    }

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
			rc = OMPI_ERR_RESOURCE_BUSY;
			goto done;
        }
    }

    if( !queued ) {
        rhdr_t ALIGN8 rhdr;
        rhdr.type = MSG_TYPE_ISEND;
        rhdr.size = size;

        if( isend_msg(peer, &rhdr, data, size) ) {
            goto done;
        }
        frag = allocfrag(RHDR_SIZE + size8);
        data = (void*)(frag + 1) + RHDR_SIZE;
        memcpy(data, txbuf, size);
    }

    rhdr_t* rhdr = (rhdr_t*)(frag + 1);
    rhdr->size = size;

    frag->msgtype = MSG_TYPE_ISEND;
    frag->size = size;
    frag->send = 0;
    frag->peer = peer;

    if( mca_btl_nc_component.async_send ) {
        push_sendq_async(peer, frag);
    }
    else {
        push_sendq_sync(peer, frag);
    }

    *descriptor = 0;

done:
    if( mca_btl_nc_component.statistics && (rc == OMPI_SUCCESS) ) {
        setstatistics(size);
    }

    return rc;
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
    int peer = endpoint->peer_smp_rank;

    // reserve is used for specific header e.g. MATCH or RDMA headers
    int hdr_size = sizeof(mca_btl_nc_hdr_t) + reserve;
    size_t max_data = *size;
    int msgsize = hdr_size + *size;
    int size8 = ((msgsize + 7) & ~7);

    frag_t* frag = allocfrag(RHDR_SIZE + size8);

    mca_btl_nc_hdr_t* hdr = (mca_btl_nc_hdr_t*)((void*)(frag + 1) + RHDR_SIZE);

    struct iovec iov;
    uint32_t iov_count = 1;
    iov.iov_len = max_data;
    iov.iov_base = (uint8_t*)hdr + hdr_size;

    frag->size = msgsize;
    frag->send = 0;
    frag->msgtype = (msgsize <= MAX_SEND_SIZE) ? MSG_TYPE_FRAG : MSG_TYPE_BLK;

    size_t sz1;
    int rc = opal_convertor_pack(convertor, &iov, &iov_count, &sz1);

    assert( rc >= 0 );
    assert( iov.iov_len == sz1 );
    assert( sz1 > 0 );

    hdr->frag = frag;
    hdr->size = msgsize;

    hdr->segment.base.seg_addr.pval = hdr + 1;
    hdr->segment.base.seg_len = reserve + max_data;

    hdr->base.des_src = &(hdr->segment.base);
    hdr->base.des_src_cnt = 1;
    hdr->base.order = MCA_BTL_NO_ORDER;
    hdr->base.des_dst = NULL;
    hdr->base.des_dst_cnt = 0;
    hdr->base.des_flags = flags;

    *size = max_data;

    if( mca_btl_nc_component.statistics ) {
        setstatistics(msgsize);
    }
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

    int peer = endpoint->peer_smp_rank;

    frag_t* frag = hdr->frag;
    frag->peer = peer;

    rhdr_t* rhdr = (rhdr_t*)(frag + 1);
    rhdr->type = MSG_TYPE_FRAG;

    if( mca_btl_nc_component.async_send ) {
        push_sendq_async(peer, frag);
    }
    else {
        push_sendq_sync(peer, frag);
    }

    return 1;
}


/**
 *
 */
void mca_btl_nc_dump(struct mca_btl_base_module_t* btl,
                     struct mca_btl_base_endpoint_t* endpoint,
                     int verbose)
{
}


void sendack(int peer, frag_t* sfrag)
{
    int size = sizeof(void*);

    // if there are already pending sends to peer, this message must be queued also
    bool queued = mca_btl_nc_component.sendqcnt[peer];

    if( !queued ) {
        rhdr_t ALIGN8 rhdr;
        rhdr.type = MSG_TYPE_ACK;
        rhdr.size = size;

        if( isend_msg(peer, &rhdr, (void*)&sfrag, size) ) {
			goto done;
        }
    }

    frag_t* frag = allocfrag(RHDR_SIZE + size);
    rhdr_t* rhdr = (rhdr_t*)(frag + 1);

    uint8_t* data = (uint8_t*)(rhdr + 1);
    *(void**)data = sfrag;

    frag->peer = peer;
    frag->size = size;
    frag->send = 0;
    frag->msgtype = MSG_TYPE_ACK;

    if( mca_btl_nc_component.async_send ) {
        push_sendq_async(peer, frag);
    }
    else {
        push_sendq_sync(peer, frag);
    }

done:
    if( mca_btl_nc_component.statistics ) {
        setstatistics(sizeof(void*));
    }
}


frag_t* allocfrag(int size)
{
    assert( (size > 0) && (size < MAX_MSG_SIZE + 1024) );

    int size8 = ((size + 7) & ~7) + sizeof(frag_t);
    // allocate frag in shared mem
    frag_t*  frag = (frag_t*)mca_btl_nc_component.shm_fragbase;

    __semlock(&mca_btl_nc_component.fraglock);

    struct msgstats_t* pstat = (struct msgstats_t*)mca_btl_nc_component.shm_stat;
    bool out_of_mem = false;
	uint32_t hicnt = 0;
	uint32_t hilvl = 0;

    for( ; ; ) {
        if( !frag->inuse && (frag->fsize >= size8) ) {
            if( out_of_mem ) {
                fprintf(stderr, "**** FRAGMENT ALLOCATED, RANK %d...\n", MY_RANK);
                fflush(stderr);
            }
            break;
        }
        if( frag->lastfrag ) {
            __semunlock(&mca_btl_nc_component.fraglock);
            if( !out_of_mem ) {
                fprintf(stderr, "****WARNING : OUT OF FRAGMENT MEMORY, RANK %d\n", MY_RANK);
                fflush(stderr);
            }
            usleep(4000000);
            frag = (frag_t*)mca_btl_nc_component.shm_fragbase;
            __semlock(&mca_btl_nc_component.fraglock);
            out_of_mem = true;
			hicnt = 0;
			hilvl = 0;
            continue;
        }
		++hicnt;
		hilvl += frag->fsize;

        frag = (frag_t*)((uint8_t*)frag + frag->fsize);
    }

    int rest = frag->fsize - size8;

    if( rest >= sizeof(frag_t) + 256 ) { // split treshold is 256
        // split frag

        frag_t* next = (frag_t*)((uint8_t*)frag + size8);

        if( !frag->lastfrag ) {
            frag_t* nx = (frag_t*)((uint8_t*)frag + frag->fsize);
            nx->prevsize = rest;
        }

        next->fsize = rest;
        next->inuse = false;
        next->lastfrag = frag->lastfrag;
        next->prevsize = size8;

        frag->lastfrag = false;
        frag->fsize = size8;
    }

    frag->inuse = true;

    __semunlock(&mca_btl_nc_component.fraglock);

   return frag;
}


void freefrag(frag_t* frag)
{
    assert( (frag->fsize > 0) && (frag->fsize < MAX_MSG_SIZE + 1024) );

    int fsize = frag->fsize;
    bool last = frag->lastfrag;

    __semlock(&mca_btl_nc_component.fraglock);

    if( !last ) {
        frag_t* next = (frag_t*)((void*)frag + fsize);
        if( !next->inuse ) {
            fsize += next->fsize;
            last = next->lastfrag;
        }
    }

    if( frag->prevsize ) {
        frag_t* prev = (frag_t*)((void*)frag - frag->prevsize);
        if( !prev->inuse ) {
            frag = prev;
            fsize += frag->fsize;
        }
    }

    frag->fsize = fsize;
    frag->inuse = false;
    frag->lastfrag = last;

    if( !last ) {
        frag_t* next = (frag_t*)((void*)frag + fsize);
        next->prevsize = fsize;
    }

    __semunlock(&mca_btl_nc_component.fraglock);
}


// sending asynchronouly with send thread
static void push_sendq_async(int peer, frag_t* frag)
{
    fifolist_t* list = &mca_btl_nc_component.pending_sends;
    frag->next = 0;

    int msgtype = frag->msgtype;

    pthread_mutex_lock(&mca_btl_nc_component.send_mutex);

	__semlock(&mca_btl_nc_component.sendlock);
         
    if( list->head ) {
        frag_t* tail = list->tail;
        tail->next = frag;
    }
    else {
        list->head = frag;
    }
    list->tail = frag;

    if( frag->msgtype == MSG_TYPE_ISEND ) {
        ++mca_btl_nc_component.sendqcnt[peer];
    }
	__semunlock(&mca_btl_nc_component.sendlock);

    pthread_mutex_unlock(&mca_btl_nc_component.send_mutex);

    // signal send thread
    pthread_cond_signal(&mca_btl_nc_component.send_cond);
}


// sending synchronouly without send thread
static void push_sendq_sync(int peer, frag_t* frag)
{
    fifolist_t* list = &mca_btl_nc_component.pending_sends;

    // peer to peer, non shared send list
    frag->next = 0;

    __semlock(&mca_btl_nc_component.sendlock);

    if( list->head ) {
        list->tail->next = frag;
    }
    else {
        list->head = frag;
    }
    list->tail = frag;

    if( frag->msgtype == MSG_TYPE_ISEND ) {
        ++mca_btl_nc_component.sendqcnt[peer];
    }

    __semunlock(&mca_btl_nc_component.sendlock);
}


static void init_ring(int peer)
{
	void* shm = mca_btl_nc_component.shm_base[peer];
	ring_t* ring = (ring_t*)shm;

	int ndx = 0;
	for( ; ; ) {
		if( __sync_bool_compare_and_swap(&ring->peer, -1, MY_RANK) ) {
			break;
		}
		++ndx;
		++ring;
	}

	mca_btl_nc_component.peer_ring_buf[peer] = shm + mca_btl_nc_component.ring_ofs + (MY_RANK * RING_SIZE);	
}


static rhdr_t* allocring(int peer, int size, int* sbit)
{
    int size8 = ((size + 7) & ~7);
    int ssize = syncsize(size8);
    int rsize = RHDR_SIZE + ssize + size8;

    void* ring_buf = mca_btl_nc_component.peer_ring_buf[peer];

    if( !ring_buf ) {
        // map target nodes rings address space
        init_ring(peer);
        ring_buf = mca_btl_nc_component.peer_ring_buf[peer];
        assert( ring_buf );
    }

    static const int RING_GUARD = RHDR_SIZE;
    uint8_t* buf = 0;

    // peer ring descriptor is in shared mem
    volatile pring_t* pr = mca_btl_nc_component.peer_ring + peer;

    uint32_t tail = pr->stail;
    uint32_t head = pr->head;

    if( head >= tail ) {
        if( head + rsize + RING_GUARD <= RING_SIZE ) {
            pr->head += rsize;
            buf = ring_buf + head;
            *sbit = (pr->sbit ^ 1);
        }
        else {
            if( rsize + RING_GUARD <= tail ) {
                assert( head + RING_GUARD <= RING_SIZE );

                // reset ring

                // toggle syncbit
                *sbit = pr->sbit;
                pr->sbit ^= 1;

                __nccopy4(ring_buf + head, MSG_TYPE_RST | pr->sbit);
                pr->head = rsize;
                buf = ring_buf;
            }
        }
    }
    else {
        if( rsize + RING_GUARD <= (tail - head) ) {
            pr->head += rsize;
            buf = ring_buf + head;
            *sbit = (pr->sbit ^ 1);
        }
    }

    return (rhdr_t*)buf;
}


static void ringbind()
{
    int rank_count = mca_btl_nc_component.num_smp_procs;

    // bind all ring address space to local node
    // it will be commited from remote
    int mynode = currNumaNode();

    struct bitmask* mask = numa_allocate_nodemask();
    numa_bitmask_clearall(mask);
    numa_bitmask_setbit(mask, mynode);

    void* rings = mca_btl_nc_component.shm_ringbase;
    int rc = syscall(__NR_mbind, (long)rings, rank_count * RING_SIZE, MPOL_BIND, (long)mask->maskp, mask->size, 0);
    assert( rc >= 0 );
    if( rc < 0 ) {
        fprintf(stderr, "WARNING : MBIND FAILED...\n");
        fflush(stderr);
    }

    numa_bitmask_free(mask);
}


static void* send_thread(void* arg)
{
    // try to find idle core on current numanode
    int cpuid = currCPU();
    if( (cpuid & 1) == 0 ) {
        int nodeid = currNumaNode();
        int newid = -1;
        if( numa_node_of_cpu(cpuid + 1) == nodeid ) {
            newid = cpuid + 1;
        }
        else
        if( numa_node_of_cpu(cpuid - 1) == nodeid ) {
            newid = cpuid - 1;
        }
        if( newid >= 0 ) {
            bind_cpu(newid);
        }
    }

    pthread_cond_t* sendq_cond = &mca_btl_nc_component.send_cond;
    pthread_condattr_t cond_attr;
    pthread_condattr_init(&cond_attr);
    pthread_cond_init(sendq_cond, &cond_attr);

    __sfence();

    mca_btl_nc_component.send_thread_active = 1;
    __sfence();

    pthread_mutex_t* sendq_mutex = &mca_btl_nc_component.send_mutex;

    int32_t* sendqcnt = mca_btl_nc_component.sendqcnt;
    volatile fifolist_t* list = &mca_btl_nc_component.pending_sends;

    volatile bool* active = &mca_btl_nc_component.send_thread_active;

    while( *active ) {

        if( !list->head && !pthread_mutex_trylock(sendq_mutex) ) {
            while( !list->head ) {
                // pthread_cond_wait() unlocks the mutex. thus you must
                // always have ownership of the mutex before invoking it
                // pthread_cond_wait() returns with the mutex locked
                pthread_cond_wait(sendq_cond, sendq_mutex);
                if( !*active ) {
                    break;
                }
            }
            pthread_mutex_unlock(sendq_mutex);
        }

        frag_t* frag = list->head;

        while( frag && __semtrylock(&mca_btl_nc_component.sendlock) ) {

			frag_t* dofree = 0;
			frag = list->head;

			if( frag ) {

				int type = frag->msgtype;
				int peer = frag->peer;
				frag_t* next = frag->next;

				if( send_msg(frag) ) {

					// takeout of send list
					list->head = next;

					if( type == MSG_TYPE_ISEND ) {
						--sendqcnt[peer];
					}

					if( type == MSG_TYPE_ISEND || type == MSG_TYPE_ACK ) {
						dofree = frag;
					}
				}
			}
			__semunlock(&mca_btl_nc_component.sendlock);

			if( dofree ) {
				freefrag(dofree);
			}

			frag = list->head;
        }
    }
    return NULL;
}


void send_sync()
{
    fifolist_t* list = &mca_btl_nc_component.pending_sends;

	frag_t* dofree = 0;

    if( __semtrylock(&mca_btl_nc_component.sendlock) ) {

		frag_t* frag = list->head;

		if( frag ) {

			// ! dont access frag after it is sent, it might be freed already
			frag_t* next = frag->next;
			int type = frag->msgtype;
			int peer = frag->peer;

			if( send_msg(frag) ) {

				list->head = next;

				if( type == MSG_TYPE_ISEND ) {
					--mca_btl_nc_component.sendqcnt[peer];
				}

				if( type == MSG_TYPE_ISEND || type == MSG_TYPE_ACK ) {
					dofree = frag;
				}
			}
		}

		__semunlock(&mca_btl_nc_component.sendlock);

		if( dofree ) {
			freefrag(dofree);
		}
	}
}


static bool send_msg(frag_t* frag)
{
    int rest = frag->size;

    int size = MAX_SEND_SIZE;
    if( rest < size ) {
        size = rest;
    }

    rhdr_t* hdr = (rhdr_t*)(frag + 1);
    const uint64_t* p = (uint64_t*)((void*)(hdr + 1) + frag->send);

    int peer = frag->peer;
    int type = frag->msgtype;

    for( ; ; ) {

        assert( size > 0 );

        int sbit;
        rhdr_t* rhdr = (rhdr_t*)allocring(peer, size, &sbit);
        if( !rhdr ) {
            break;
        }

        uint64_t* q = (uint64_t*)(rhdr + 1);

        hdr->type = (type | sbit);
        hdr->size = size;

        frag->send += size;
        frag->size -= size;

        int size8 = ((size + 7) & ~7);
        int n = (size8 >> 3);
        int k = n;
        if( k > 32 ) {
            k = 32;
        }

        uint64_t b = 0;

        for( int i = 0; i < k; i++ ) {
            uint64_t z = *p++;
            b <<= 1;
            b |= (z & 1);

            z >>= 1;
            z <<= 1;
            z |= sbit;
            __asm__ __volatile__ (
                "movq %0, %%rax\n"
                "movnti %%rax, (%1)\n"
                :: "r" (z), "r" (q) : "rax", "memory");
            ++q;
        }
        hdr->sbits = (uint32_t)b;

        __asm__ __volatile__ (
            "movq (%0), %%rax\n"
            "movnti %%rax, (%1)\n"
            :: "r" (hdr), "r" (rhdr) : "rax", "memory");

        n -= k;

        if( n ) {
            scopy(q, p, n << 3, sbit);
            p += n;
        }

        __asm__ __volatile__ ("sfence\n");

        rest -= size;
        assert( rest >= 0 );

        if( rest == 0 ) {
            break;
        }

        if( rest <= size ) {
            size = rest;
        }
    }

    return (rest == 0);
}


static bool isend_msg(int node, rhdr_t* hdr, void* buf, int size)
{
    assert( size <= MAX_SEND_SIZE );

    const uint64_t* p = (uint64_t*)buf;

    int size8 = ((size + 7) & ~7);
    int n = (size8 >> 3);

	__semlock(&mca_btl_nc_component.sendlock);

    int sbit;
    rhdr_t* rhdr = (rhdr_t*)allocring(node, size, &sbit);
    if( !rhdr ) {
		__semunlock(&mca_btl_nc_component.sendlock);
        return false;
    }

    uint64_t* q = (uint64_t*)(rhdr + 1);

    hdr->type |= sbit;

    int k = n;
    if( k > 32 ) {
        k = 32;
    }

    uint64_t b = 0;

    for( int i = 0; i < k; i++ ) {
        uint64_t z = *p++;
        b <<= 1;
        b |= (z & 1);

        z >>= 1;
        z <<= 1;
        z |= sbit;
        __asm__ __volatile__ (
            "movq %0, %%rax\n"
            "movnti %%rax, (%1)\n"
            :: "r" (z), "r" (q) : "rax", "memory");
        ++q;
    }
    hdr->sbits = (uint32_t)b;

    __asm__ __volatile__ (
        "movq (%0), %%rax\n"
        "movnti %%rax, (%1)\n"
        :: "r" (hdr), "r" (rhdr) : "rax", "memory");

    n -= k;

    if( n ) {
        scopy(q, p, n << 3, sbit);
    }

    __asm__ __volatile__ ("sfence\n");

	__semunlock(&mca_btl_nc_component.sendlock);

    return true;
}


static void scopy(void* dst, const void* src, int size, int sbit)
{
    assert( ((uint64_t)dst & 0x3) == 0 );
    assert( ((uint64_t)src & 0x3) == 0 );
    assert( (size > 0) && ((size & 7) == 0) );
    assert( (sbit == 0) || (sbit == 1) );

    int size8 = ((size + 7) & ~7);
    int n = (size8 >> 3);

    const uint64_t* p = (uint64_t*)src;
    uint64_t* q = (uint64_t*)dst;

    while( n ) {

        int k = n;
        if( k > 63 ) {
            k = 63;
        }

        uint64_t b = 0;

        for( int i = 0; i < k; i++ ) {
            uint64_t z = *p++;
            b <<= 1;
            b |= (z & 1);

            z >>= 1;
            z <<= 1;
            z |= sbit;
            __asm__ __volatile__ (
                "movq %0, %%rax\n"
                "movnti %%rax, (%1)\n"
                :: "r" (z), "r" (q) : "rax", "memory");
            ++q;
        }

        b <<= 1;
        b |= sbit;
        __asm__ __volatile__ (
            "movq %0, %%rax\n"
            "movnti %%rax, (%1)\n"
            :: "r" (b), "r" (q) : "rax", "memory");
        ++q;

        n -= k;
    }
}

