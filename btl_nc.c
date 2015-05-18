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
static void sbitset256(void* sbits, void* data, int size8, int sbit);
static void sbitset(void* sbits, void* data, int size8, int sbit);
static uint8_t* allocring(int peer_node, int size8, int* sbit);
static bool send_msg(frag_t* frag);
static void push_peerq(int peer, frag_t* frag);
static void push_sendq(int peer, frag_t* frag);

frag_t* allocfrag(int size);
void freefrag(frag_t* frag);
void INThandler(int sig);


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


static void* map_shm(const int node, const int shmsize)
{
    char* nc_ctl_file;

    if( node >= 0 ) {
        if( asprintf(&nc_ctl_file, "%s"OPAL_PATH_SEP"nc_btl_module.%s.%d",
                     orte_process_info.job_session_dir,
                     orte_process_info.nodename,
                     node) < 0 ) {
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

    if( MAP_FAILED == shm ) {
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

    if (OMPI_SUCCESS != return_code)
        goto CLEANUP;

CLEANUP:
    return return_code;
}


static int numa_dist(int i, int j)
{
    sysctxt_t* sysctxt = mca_btl_nc_component.sysctxt;
    return sysctxt->numadist[i * sysctxt->max_nodes + j];
}


static int add_to_group()
{
    size_t pagesize = sysconf(_SC_PAGESIZE);
    int max_cpus = numa_num_configured_cpus();
    int max_nodes = numa_num_configured_nodes();

    size_t noderecsize = ((sizeof(node_t) + pagesize - 1) & ~(pagesize - 1));
    size_t syssize = ((sizeof(sysctxt_t) + pagesize - 1) & ~(pagesize - 1));
    size_t shmsize = syssize + (max_nodes * noderecsize);

    sysctxt_t* sysctxt = (sysctxt_t*)map_shm(-1, shmsize);
    if( !sysctxt ) {
        return -1;
    }
    mca_btl_nc_component.sysctxt = sysctxt;
    sysctxt->max_nodes = max_nodes;

    int cpuid = currCPU();
    sysctxt->cpuid[MY_RANK] = cpuid + 1;
    int currnode = currNumaNode();

    void* nodedesc0 = (void*)sysctxt + syssize;
    int32_t* numadist = sysctxt->numadist;
    int32_t* numanode = sysctxt->numanode;

    __semlock(&sysctxt->lock);

    int rank_count = ++sysctxt->rank_count;

    if( numadist[0] == 0 ) {
        // cache numa info
        for( int i = 0; i < max_nodes; i++ ) {
            for( int j = 0; j < max_nodes; j++ ) {
                numadist[i * max_nodes + j] = numa_distance(i, j);
            }
        }

        for( int i = 0; i < max_cpus; i++ ) {
            numanode[i] = numa_node_of_cpu(i);
        }
    }

    int group = -1;
    if( mca_btl_nc_component.grp_numa_dist <= INTRA_GROUP_NUMA_DIST ) {
        group = currnode;
    }
    else {
        for( int i = 0; i < max_nodes; i++ ) {
            if( numadist[currnode * max_nodes + i] <= mca_btl_nc_component.grp_numa_dist ) {
                if( sysctxt->group[i] - 1 >= 0 ) {
                    group = sysctxt->group[i] - 1;
                    break;
                }
            }
        }
        if( group < 0 ) {
            group = sysctxt->node_count++;
            sysctxt->group[currnode] = group + 1;
        }
    }

    node_t* nodedesc = (node_t*)(nodedesc0 + group * noderecsize);
    int32_t cpuindex = nodedesc->ndxmax++;

    __semunlock(&sysctxt->lock);

    mca_btl_nc_component.group = group;
    mca_btl_nc_component.cpuindex = cpuindex;
    sysctxt->map[MY_RANK] = ((((uint32_t)group) << 16) | cpuindex);

    return group;
}


static int nc_btl_first_time_init(mca_btl_nc_t* nc_btl, int n)
{
    assert( sizeof(frag_t) == FRAG_SIZE );

    size_t pagesize = sysconf(_SC_PAGESIZE);
    int max_cpus = numa_num_configured_cpus();
    int max_nodes = numa_num_configured_nodes();

    char* ev = getenv("NC_GROUP_NUMA_DIST");
    int grp_numa_dist = INTRA_GROUP_NUMA_DIST;
    if( ev && (sscanf(ev, "%d", &grp_numa_dist) == 1) ) {
        if( grp_numa_dist <= 0 || grp_numa_dist >= 100 ) {
            if( MY_RANK == 0 ) {
                fprintf(stderr, "NC_GROUP_NUMA_DIST = %d INAVLID, USING DEFAULT VALUE = 10\n",
                        grp_numa_dist);
                fflush(stderr);
            }
            grp_numa_dist = INTRA_GROUP_NUMA_DIST;
        }
    }
    mca_btl_nc_component.grp_numa_dist = grp_numa_dist;

    size_t noderecsize = ((sizeof(node_t) + pagesize - 1) & ~(pagesize - 1));
    size_t syssize = ((sizeof(sysctxt_t) + pagesize - 1) & ~(pagesize - 1));
    size_t shmsize = syssize + (max_nodes * noderecsize);

    int node = add_to_group();
    if( node < 0 ) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    sysctxt_t* sysctxt = mca_btl_nc_component.sysctxt;
    sysctxt->num_smp_procs = n;
    mca_btl_nc_component.map = sysctxt->map;

    int cpuindex = mca_btl_nc_component.cpuindex;

    void* node0 = (void*)sysctxt + syssize;
    mca_btl_nc_component.peer_node = (node_t**)malloc(max_nodes * sizeof(void*));
    for( int i = 0; i < max_nodes; i++ ) {
        mca_btl_nc_component.peer_node[i] = (node_t*)(node0 + i * noderecsize);
    }

    mca_btl_nc_component.nodedesc = mca_btl_nc_component.peer_node[node];
    volatile node_t* nodedesc = mca_btl_nc_component.nodedesc;

    // init pointers to send tails in shared mem
    // these pointers will be used by the receiver to reset the send tail on the sender side
    mca_btl_nc_component.peer_stail = (uint32_t**)malloc(max_nodes * sizeof(void*));
    for( int i = 0; i < max_nodes; i++ ) {
        mca_btl_nc_component.peer_stail[i] = mca_btl_nc_component.peer_node[i]->stail;
    }

    mca_btl_nc_component.stail = (uint32_t**)malloc(max_nodes * sizeof(void*));
    uint32_t* stail = mca_btl_nc_component.peer_stail[node];
    for( int i = 0; i < max_nodes; i++ ) {
        // use one cache line per counter
        mca_btl_nc_component.stail[i] = stail + i;
    }

    /*
      ring descriptors
      peer ring descriptors
      pending sends lists
      input lists
      rings
      fragments
    */

    shmsize = max_nodes * sizeof(ring_t);
    shmsize += max_nodes * sizeof(pring_t);
    shmsize += sizeof(fifolist_t);
    shmsize += n * 16 * sizeof(int32_t);
    shmsize += max_nodes * sizeof(fifolist_t);
    shmsize = (shmsize + pagesize - 1) & ~(pagesize - 1);

    size_t ring_ofs = shmsize;

    shmsize += (size_t)max_nodes * (size_t)RING_SIZE;

    size_t frag_ofs = shmsize;

    shmsize += MAX_SIZE_FRAGS;
    shmsize = (shmsize + pagesize - 1) & ~(pagesize - 1);

    // map local mem
    void* shmbase = map_shm(node, shmsize);
    assert( shmbase );

    mca_btl_nc_component.shmsize = shmsize;
    mca_btl_nc_component.shm_base = shmbase;
    mca_btl_nc_component.ring_ofs = ring_ofs;
    mca_btl_nc_component.frag_ofs = frag_ofs;
    mca_btl_nc_component.shm_ringbase = shmbase + ring_ofs;
    mca_btl_nc_component.shm_fragbase = shmbase + frag_ofs;

    // init list of pointers to local peer ring descriptors
    mca_btl_nc_component.ring = (ring_t*)shmbase;

    // init list of pointers to local peer ring descriptors
    mca_btl_nc_component.peer_ring = (pring_t*)(shmbase + max_nodes * sizeof(ring_t));

    mca_btl_nc_component.peer_ring_buf = (void**)malloc(max_nodes * sizeof(void*));
    memset(mca_btl_nc_component.peer_ring_buf, 0, max_nodes * sizeof(void*));

    mca_btl_nc_component.pending_sends = (fifolist_t*)(
        shmbase + max_nodes * sizeof(ring_t) + max_nodes * sizeof(pring_t));

    // local input queue
    mca_btl_nc_component.inq = shmbase
                               + max_nodes * sizeof(ring_t)
                               + max_nodes * sizeof(pring_t)
                               + sizeof(fifolist_t)
                               + n * 16 * sizeof(int32_t);

    mca_btl_nc_component.myinq = mca_btl_nc_component.inq + cpuindex;

    mca_btl_nc_component.sendqcnt = shmbase
                                    + max_nodes * sizeof(ring_t)
                                    + max_nodes * sizeof(pring_t)
                                    + sizeof(fifolist_t);

    if( cpuindex == 0 ) {
        pthread_create(&mca_btl_nc_component.sendthread, 0, &send_thread, 0);
    }

    ev = getenv("NCSTAT");
    mca_btl_nc_component.statistics = (ev && (!strcasecmp(ev, "yes") || !strcasecmp(ev, "true") || !strcasecmp(ev, "1")));
    if( mca_btl_nc_component.statistics ) {
        mca_btl_nc_component.statistics = createstat(n);
    }

    // wait until send thread ready
    while( !nodedesc->active );

    // offset for local processes to local fragments
    mca_btl_nc_component.shm_ofs = shmbase - nodedesc->shm_base;
/*
  int32_t readycnt = lockedAdd(&sysctxt->readycnt, 1);

  fprintf(stderr, "READY %d, RANK %d, TOTAL %d, PID %d, GROUP %d, CPUID %d, CPU INDEX %d, SHMSIZE %d MB\n",
  readycnt,
  MY_RANK,
  sysctxt->rank_count,
  getpid(),
  mca_btl_nc_component.group,
  sysctxt->cpuid[MY_RANK] - 1,
  mca_btl_nc_component.cpuindex,
  (int)(shmsize >> 20));
  fflush(stderr);
  }
*/
    if( MY_RANK == 0 ) {
        printf("************USING NC-BTL 1.8.3************\n");
    }

    signal(SIGUSR1, INThandler);

//    assert( lzo_init() == LZO_E_OK );

    /* set flag indicating btl has been inited */
    nc_btl->btl_inited = true;
    __mfence();

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
    mca_btl_nc_component.nodedesc->active = 0;
    pthread_cond_signal(&mca_btl_nc_component.nodedesc->send_cond);

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
    int peer = endpoint->peer_smp_rank;
    uint32_t dst = mca_btl_nc_component.map[peer];
    int peer_node = (dst >> 16);

    uint32_t size8 = ((sizeof(mca_btl_nc_hdr_t) + size + 7) & ~7);
    mca_btl_nc_hdr_t* hdr;
    frag_t* frag;

    if( peer_node != mca_btl_nc_component.group ) {
        frag = allocfrag(sizeof(rhdr_t) + size8);
        hdr = (mca_btl_nc_hdr_t*)((void*)(frag + 1) + sizeof(rhdr_t));
    }
    else {
        frag = allocfrag(size8);
        hdr = (mca_btl_nc_hdr_t*)(frag + 1);
    }

    if( !frag ) {
        return 0;
    }

    hdr->self = frag;
    hdr->reserve = size;
    hdr->base.des_src = &hdr->segment.base;
    hdr->base.des_src_cnt = 1;
    hdr->base.des_dst = &hdr->segment.base;
    hdr->base.des_dst_cnt = 1;
    hdr->segment.base.seg_len = size;
    hdr->base.des_flags = flags;
    hdr->segment.base.seg_addr.pval = hdr + 1;

    frag->size = sizeof(mca_btl_nc_hdr_t) + size;
    frag->ofs = 0;
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
    int peer = endpoint->peer_smp_rank;
    uint32_t dst = mca_btl_nc_component.map[peer];
    int peer_node = (dst >> 16);

    if( peer_node == mca_btl_nc_component.group ) {
        int size = payload_size + header_size;
        int size8 = ((size + 7) & ~7);

        frag_t* frag = allocfrag(size8);
        if( !frag ) {
            // upper layers will call progress() first and than try sending again
            return OMPI_ERR_RESOURCE_BUSY;
        }

        frag->msgtype = MSG_TYPE_ISEND;
        frag->size = size;
        void* data = frag + 1;

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

        push_peerq(dst & 0xffff, frag);

        return OMPI_SUCCESS;
    }

    int size = payload_size + header_size;
    int size8 = ((size + 7) & ~7);
    int ssize = syncsize(size8);
    int rsize = sizeof(rhdr_t) + size8;

    // if there are already pending sends to peer, this message must be queued also
    bool queued = (bool)mca_btl_nc_component.sendqcnt[peer << 4];

    int sbit;
    uint8_t* rbuf;
    frag_t* frag;
    uint8_t buf[sizeof(frag_t) + rsize + ssize] ALIGN8;

    if( !queued ) {
        rbuf = allocring(peer_node, rsize + ssize, &sbit);
        if( !rbuf ) {
            queued = true;
        }
    }

    if( queued ) {
        frag = allocfrag(rsize);
        if( !frag ) {
            // upper layers will call progress() first and than try sending again
            assert( false );
            return OMPI_ERR_RESOURCE_BUSY;
        }
    }
    else {
        frag = (frag_t*)buf;
    }

    rhdr_t* rhdr = (rhdr_t*)(frag + 1);
    void* data = rhdr + 1;

    rhdr->dst_ndx = (dst & 0xffff);
    rhdr->rsize = ((rsize + ssize) >> 2);
    rhdr->pad8 = (size8 - size);

    assert( rhdr->rsize > 0 );

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
            assert( false );
            return OMPI_ERR_RESOURCE_BUSY;
        }
    }

    if( queued ) {
        frag->node = peer_node;
        frag->size = size;
        frag->ofs = 0;
        frag->msgtype = MSG_TYPE_ISEND;
        frag->peer = peer;

        push_sendq(peer, frag);

        return OMPI_SUCCESS;
    }

    rhdr->type = (MSG_TYPE_ISEND | sbit);

    if( size8 <= SHDR ) {
        void* sbits = (void*)&rhdr->sbits;
        sbitset256(sbits, data, size8, sbit);
        nccopy(rbuf, rhdr, rsize);
    }
    else {
        uint64_t sbits[size8 >> 3] ALIGN8;
        sbitset(sbits, data, size8, sbit);

        void* dst = rbuf;
        __nccopy8(dst, rhdr);

        dst += sizeof(rhdr_t);
        nccopy(dst, sbits, ssize);

        dst += ssize;
        nccopy(dst, data, size8);
    }

    __sfence();
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
    int peer = endpoint->peer_smp_rank;
    uint32_t dst = mca_btl_nc_component.map[peer];
    int peer_node = (dst >> 16);

    // reserve is used for specific header e.g. MATCH or RDMA headers
    int hdr_size = sizeof(mca_btl_nc_hdr_t) + reserve;
    size_t max_data = *size;
    int msgsize = hdr_size + *size;

    int size8 = ((msgsize + 7) & ~7);
    int rsize = sizeof(rhdr_t) + size8;

    frag_t* frag = allocfrag(rsize);

    mca_btl_nc_hdr_t* hdr = (mca_btl_nc_hdr_t*)(frag + 1);
    if( peer_node != mca_btl_nc_component.group ) {
        hdr = (mca_btl_nc_hdr_t*)((uint8_t*)hdr + sizeof(rhdr_t));
    }

    struct iovec iov;
    uint32_t iov_count = 1;
    iov.iov_len = max_data;
    iov.iov_base = (uint8_t*)hdr + hdr_size;

    frag->size = msgsize;
    frag->ofs = 0;
    frag->msgtype = (msgsize > MAX_SEND_SIZE) ? MSG_TYPE_BLK : MSG_TYPE_FRAG;

    size_t sz1;
    int rc = opal_convertor_pack(convertor, &iov, &iov_count, &sz1);

    assert( rc >= 0 );
    assert( iov.iov_len == sz1 );
    assert( sz1 > 0 );

    hdr->self = frag;
    hdr->reserve = reserve;

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

    frag_t* frag = hdr->self;

    if( MCA_BTL_DES_SEND_ALWAYS_CALLBACK & hdr->base.des_flags ) {
        // create ack hdr (contains a copy of header for use in ack message)
        int hdr_size = sizeof(mca_btl_nc_hdr_t) + hdr->reserve;
        frag_t* ackfrag = allocfrag(hdr_size);

        mca_btl_nc_hdr_t* ackhdr = (mca_btl_nc_hdr_t*)(ackfrag + 1);

        hdr->self = ackfrag;
        memcpy(ackhdr, hdr, hdr_size);

        ackhdr->segment.base.seg_addr.pval = ackhdr + 1;
        ackhdr->base.des_src = &(ackhdr->segment.base);
    }

    int peer = endpoint->peer_smp_rank;
    uint32_t dst = mca_btl_nc_component.map[peer];
    int peer_node = (dst >> 16);

    if( peer_node != mca_btl_nc_component.group ) {
        frag->node = peer_node;
        frag->peer = peer;

        rhdr_t* rhdr = (rhdr_t*)(frag + 1);
        rhdr->type = MSG_TYPE_FRAG;
        rhdr->dst_ndx = (dst & 0xffff);

        push_sendq(peer, frag);
    }
    else {
        frag->msgtype = MSG_TYPE_FRAG;
        push_peerq(dst & 0xffff, frag);
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


static bool send_msg(frag_t* frag)
{
    int rest = frag->size;
    int ofs = frag->ofs;

    int size = MAX_SEND_SIZE;
    if( rest < size ) {
        size = rest;
    }

    rhdr_t* rhdr = (rhdr_t*)(frag + 1);
    void* data = (void*)(rhdr + 1) + ofs;

    // if message is multi part, then all but last parts are larger
    // then MAX_SEND_SIZE and rhdr->sbits is unused, use it for total message size
    rhdr->sbits = frag->size;

    int size8 = ((size + 7) & ~7);
    int ssize = syncsize(size8);
    int rsize = sizeof(rhdr_t) + ssize + size8;

    for( ; ; ) {
        int sbit;
        uint8_t* rbuf = allocring(frag->node, rsize, &sbit);
        if( !rbuf ) {
            break;
        }

        rhdr->type = (frag->msgtype | sbit);
        rhdr->rsize = (rsize >> 2);
        rhdr->pad8 = (size8 - size);
        assert( size8 - size <= 7 );
        assert( rhdr->rsize > 0 );

        if( size8 <= SHDR ) {
            void* sbits = (void*)&rhdr->sbits;
            sbitset256(sbits, data, size8, sbit);

            if( ofs == 0 ) {
                nccopy(rbuf, rhdr, rsize);
            }
            else {
                __nccopy8(rbuf, rhdr);
                nccopy(rbuf + sizeof(rhdr_t), data, size8);
            }
        }
        else {
            uint64_t sbits[size8 >> 3] ALIGN8;
            sbitset(sbits, data, size8, sbit);

            void* dst = rbuf;
            __nccopy8(dst, rhdr);

            dst += sizeof(rhdr_t);
            nccopy(dst, sbits, ssize);

            dst += ssize;
            nccopy(dst, data, size8);
        }

        __sfence();

        ofs += size;
        data += size;
        rest -= size;
        assert( rest >= 0 );

        if( rest == 0 ) {
            break;
        }

        if( rest <= size ) {
            size = rest;
            size8 = ((size + 7) & ~7);
            ssize = syncsize(size8);
            rsize = sizeof(rhdr_t) + ssize + size8;
            frag->msgtype = MSG_TYPE_BLKN;
        }
    }
    frag->ofs = ofs;
    frag->size = rest;

    return (rest == 0);
}


void sendack(int peer, void* hdr)
{
    uint32_t dst = mca_btl_nc_component.map[peer];
    int peer_node = (dst >> 16);

    if( peer_node == mca_btl_nc_component.group ) {
        frag_t* frag = allocfrag(sizeof(void*));
        frag->msgtype = MSG_TYPE_ACK;
        frag->size = sizeof(void*);
        *(void**)(frag + 1) = hdr;

        push_peerq(dst & 0xffff, frag);
        return;
    }

    uint32_t rsize = sizeof(rhdr_t) + sizeof(void*);

    int sbit;

    void* rbuf = allocring(peer_node, rsize, &sbit);

    if( !rbuf ) {
        frag_t* frag = allocfrag(rsize);
        rhdr_t* rhdr = (rhdr_t*)(frag + 1);

        rhdr->dst_ndx = (dst & 0xffff);
        rhdr->sbits = 0;

        uint8_t* data = (uint8_t*)(rhdr + 1);
        *(void**)data = hdr;

        frag->node = peer_node;
        frag->peer = peer;
        frag->size = sizeof(void*);
        frag->ofs = 0;
        frag->rsize = rsize;
        frag->msgtype = MSG_TYPE_ACK;

        push_sendq(peer, frag);
        return;
    }

    uint8_t buf[rsize] ALIGN8;
    rhdr_t* rhdr = (rhdr_t*)buf;

    rhdr->type = (MSG_TYPE_ACK | sbit);
    rhdr->dst_ndx = (dst & 0xffff);
    rhdr->rsize = (rsize >> 2);
    rhdr->sbits = 0;
    rhdr->pad8 = 0;

    assert( rhdr->rsize > 0 );

    void* data = rhdr + 1;
    *(void**)data = hdr;

    void* sbits = (void*)&rhdr->sbits;
    sbitset256(sbits, data, sizeof(void*), sbit);

    nccopy(rbuf, buf, rsize);

    __sfence();

    if( mca_btl_nc_component.statistics ) {
        setstatistics(sizeof(void*));
    }
}


static void nccopy(void* to, const void* from, int n)
{
    assert( n > 0 );
    assert( (n & 0x7) == 0 );
    assert( (((uint64_t)from) & 0x7) == 0 );
    assert( (((uint64_t)to) & 0x7) == 0 );

    __asm__ __volatile__ (
        "movl %0, %%ecx\n"
        "shr  $3, %%ecx\n"
        "movq %1, %%rsi\n"
        "movq %2, %%rdi\n"
        "1:\n"
        "movq (%%rsi), %%rax\n"
        "movnti %%rax, (%%rdi)\n"
        "addq $8, %%rsi\n"
        "addq $8, %%rdi\n"
        "loop 1b\n"
        : : "r" (n), "r" (from), "r" (to) : "ecx", "rax", "rsi", "rdi", "memory");
}


frag_t* allocfrag(int size)
{
    // allocate frag in shared mem
    size += sizeof(frag_t);

    frag_t* frag = (frag_t*)mca_btl_nc_component.shm_fragbase;

    node_t* nodedesc = mca_btl_nc_component.nodedesc;
    __semlock(&nodedesc->fraglock);

    for( ; ; ) {
        if( !frag->inuse && (frag->fsize >= size) ) {
            break;
        }
        if( frag->lastfrag ) {
            fprintf(stderr, "****WARNING : OUT OF FRAGMENT MEMORY, RANK %d\n", MY_RANK);
            fflush(stderr);
            __semunlock(&nodedesc->fraglock);

            frag = (frag_t*)mca_btl_nc_component.shm_fragbase;
            usleep(1000000);
            __semlock(&nodedesc->fraglock);
            continue;
        }

        frag = (frag_t*)((uint8_t*)frag + frag->fsize);
    }

    int rest = frag->fsize - size;

    if( rest >= sizeof(frag_t) + 256 ) { // split treshold is 256
        // split frag

        frag_t* next = (frag_t*)((uint8_t*)frag + size);

        if( !frag->lastfrag ) {
            frag_t* nx = (frag_t*)((uint8_t*)frag + frag->fsize);
            nx->prevsize = rest;
        }

        next->fsize = rest;
        next->inuse = false;
        next->lastfrag = frag->lastfrag;
        next->prevsize = size;

        frag->lastfrag = false;
        frag->fsize = size;
    }

    frag->inuse = true;

    __semunlock(&nodedesc->fraglock);

    return frag;
}


void freefrag(frag_t* frag)
{
    int fsize = frag->fsize;
    bool last = frag->lastfrag;

    node_t* nodedesc = mca_btl_nc_component.nodedesc;

    __semlock(&nodedesc->fraglock);

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

    __semunlock(&nodedesc->fraglock);
}


// append to fifo list, multiple producers, single consumer
static void push_peerq(int peer, frag_t* frag)
{
    frag->next = 0;
    frag = (frag_t*)NODEADDR(frag);

    fifolist_t* list = mca_btl_nc_component.inq + peer;

    __semlock(&list->lock);

    if( list->head ) {
        frag_t* tail = (frag_t*)PROCADDR(list->tail);
        tail->next = frag;
    }
    else {
        list->head = frag;
    }
    list->tail = frag;

    __semunlock(&list->lock);
}


static void push_sendq(int peer, frag_t* frag)
{
    node_t* nodedesc = mca_btl_nc_component.nodedesc;
    fifolist_t* list = mca_btl_nc_component.pending_sends;

    frag->next = 0;
    frag = (frag_t*)NODEADDR(frag);

    pthread_mutex_lock(&nodedesc->send_mutex);

    if( list->head ) {
        frag_t* tail = (frag_t*)PROCADDR(list->tail);
        tail->next = frag;
    }
    else {
        list->head = frag;
    }
    list->tail = frag;

    ++mca_btl_nc_component.sendqcnt[peer << 4];

    pthread_mutex_unlock(&nodedesc->send_mutex);

    pthread_cond_signal(&nodedesc->send_cond);
}



static void init_ring(int peer_node)
{
    int loc_node = mca_btl_nc_component.group;
//    assert( peer_node != loc_node );

    void* ring_addr = map_shm(peer_node, mca_btl_nc_component.shmsize);
    assert( ring_addr );
    ring_addr += mca_btl_nc_component.ring_ofs + (loc_node * RING_SIZE);

    volatile pring_t* pr = mca_btl_nc_component.peer_ring + peer_node;
    if( !pr->commited ) {

        __semlock(&pr->lock);

        if( !pr->commited ) {
            pr->commited = true;

            node_t* peer_nodedesc = mca_btl_nc_component.peer_node[peer_node];
            lockedAdd((int32_t*)&(peer_nodedesc->inuse[loc_node << 1]), 1);
            __sfence();

            lockedAdd(&peer_nodedesc->ring_cnt, 1);
        }
        __semunlock(&pr->lock);
    }

    mca_btl_nc_component.peer_ring_buf[peer_node] = ring_addr;
}


static uint8_t* allocring(int peer_node, int size8, int* sbit)
{
    assert( (size8 & 7) == 0 );
    assert( size8 < RING_SIZE / 2 );

    void* ring_buf = mca_btl_nc_component.peer_ring_buf[peer_node];

    if( !ring_buf ) {
        // map target nodes rings address space
        init_ring(peer_node);
        ring_buf = mca_btl_nc_component.peer_ring_buf[peer_node];
        assert( ring_buf );
    }

    static const int RING_GUARD = 8;
    uint8_t* buf = 0;

    // peer ring descriptor is in shared mem
    volatile pring_t* pr = mca_btl_nc_component.peer_ring + peer_node;

    __semlock(&pr->lock);

    uint32_t head = pr->head;
    uint32_t tail = *(mca_btl_nc_component.stail[peer_node]);

    if( head >= tail ) {
        if( head + size8 + RING_GUARD <= RING_SIZE ) {
            pr->head += size8;
            buf = ring_buf + head;
            *sbit = (pr->sbit ^ 1);
        }
        else {
            if( size8 + RING_GUARD <= tail ) {
                assert( head + RING_GUARD <= RING_SIZE );

                // reset ring

                // toggle syncbit
                *sbit = pr->sbit;
                pr->sbit ^= 1;

                __nccopy4(ring_buf + head, MSG_TYPE_RST | pr->sbit);
                pr->head = size8;
                buf = ring_buf;
            }
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


static void sbitset256(void* sbits, void* data, int size8, int sbit)
{
    // pointer to workload
    uint64_t* p = (uint64_t*)data;
    assert( ((uint64_t)p & 0x7) == 0 );

    int n = (size8 >> 3);

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


static void sbitset(void* sbits, void* data, int size8, int sbit)
{
    // pointer to workload
    uint64_t* p = (uint64_t*)data;
    assert( ((uint64_t)p & 0x7) == 0 );

    int n = (size8 >> 3);

    // pointer to sync bits
    uint64_t* q = (uint64_t*)sbits;
    assert( ((uint64_t)q & 0x7) == 0 );

    uint64_t b = 0;
    int k = 0;

    if( sbit ) {
        for( int i = 0; i < n; i++ ) {

            b |= ((*p) & 1);
            b <<= 1;
            (*p) |= 1;
            ++p;

            if( ++k < 63 ) {
                continue;
            }
            *q++ = (b | 1);
            b = 0;
            k = 0;
        }
        if( k ) {
            *q = (b | 1);
        }
    }
    else {
        for( int i = 0; i < n; i++ ) {

            b |= ((*p) & 1);
            b <<= 1;
            (*p) &= ~1;
            ++p;

            if( ++k < 63 ) {
                continue;
            }
            *q++ = b;
            b = 0;
            k = 0;
        }
        if( k ) {
            *q = b;
        }
    }
}


static void ringbind()
{
    sysctxt_t* sysctxt = mca_btl_nc_component.sysctxt;
    int max_nodes = sysctxt->max_nodes;

    // bind all ring address space to local node
    // it will be commited from remote
    int target_node = currNumaNode();

    struct bitmask* mask = numa_allocate_nodemask();
    numa_bitmask_clearall(mask);
    numa_bitmask_setbit(mask, target_node);

    void* rings = mca_btl_nc_component.shm_base + mca_btl_nc_component.ring_ofs;
    int rc = syscall(__NR_mbind, (long)rings, max_nodes * RING_SIZE, MPOL_BIND, (long)mask->maskp, mask->size, 0);
    assert( rc >= 0 );
    if( rc < 0 ) {
        fprintf(stderr, "WARNING : MBIND FAILED...\n");
        fflush(stderr);
    }

    numa_bitmask_free(mask);
}


static void memclear(void* to, int n)
{
    assert( n > 0 );
    assert( (n & 0x7) == 0 );
    assert( (((uint64_t)to) & 0x7) == 0 );

    __asm__ __volatile__ (
        "movl %0, %%ecx\n"
        "shr  $3, %%ecx\n"
        "movq %1, %%rdi\n"
        "movq $0, %%rax\n"
        "1:\n"
        "movnti %%rax, (%%rdi)\n"
        "addq $8, %%rdi\n"
        "loop 1b\n"
        : : "r" (n), "r" (to) : "rax", "ecx", "rdi", "memory");
}


static void* send_thread(void* arg)
{
    node_t* nodedesc = mca_btl_nc_component.nodedesc;

    volatile sysctxt_t* sysctxt = mca_btl_nc_component.sysctxt;
    int max_nodes = sysctxt->max_nodes;

    // init node structure, dont clear last structure member cpuindex
    memset(nodedesc, 0, offsetof(node_t, ndxmax));
    memset(mca_btl_nc_component.shm_base, 0, mca_btl_nc_component.ring_ofs);

    ringbind();

    nodedesc->shm_base = mca_btl_nc_component.shm_base;
    nodedesc->sendqcnt = mca_btl_nc_component.sendqcnt;
    int32_t* sendqcnt = nodedesc->sendqcnt;

    nodedesc->shm_frags = nodedesc->shm_base + mca_btl_nc_component.frag_ofs;
    frag_t* frag = (frag_t*)nodedesc->shm_frags;
    frag->inuse = false;
    frag->prevsize = 0;
    frag->fsize = MAX_SIZE_FRAGS;
    frag->lastfrag = true;

    volatile fifolist_t* list = mca_btl_nc_component.pending_sends;

    while( sysctxt->rank_count < sysctxt->num_smp_procs );

    pthread_condattr_t cond_attr;
    pthread_condattr_init(&cond_attr);
    pthread_condattr_setpshared(&cond_attr, PTHREAD_PROCESS_SHARED);
    pthread_cond_init(&nodedesc->send_cond, &cond_attr);

    pthread_mutex_t* sendq_mutex = &nodedesc->send_mutex;
    pthread_mutexattr_t mutex_attr;
    pthread_mutexattr_init(&mutex_attr);
    pthread_mutexattr_setpshared(&mutex_attr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(sendq_mutex, &mutex_attr);

    __sfence();
    nodedesc->active = 1;
    __sfence();

    bool* skip = (bool*)malloc(max_nodes * sizeof(bool));
    bool skipped = true;

    while( nodedesc->active ) {
        if( skipped ) {
            memset(skip, 0, max_nodes * sizeof(bool));
            skipped = false;
        }

        if( !list->head && !pthread_mutex_trylock(sendq_mutex) ) {
            while( !list->head ) {
                // pthread_cond_wait() unlocks the mutex. thus you must
                // always have ownership of the mutex before invoking it
                // pthread_cond_wait() returns with the mutex locked
                pthread_cond_wait(&nodedesc->send_cond, &nodedesc->send_mutex);
                if( !nodedesc->active ) {
                    return NULL;
                }
            }
            pthread_mutex_unlock(sendq_mutex);
        }

        frag_t* frag = list->head;
        frag_t* prev = 0;

        // send all pending sends
        // if message can not be send, skip traget for all
        // messages to this target node in this round
        while( frag ) {
            frag_t* next = frag->next;
            int peer_node = frag->node;

            if( !skip[peer_node] ) {

                if( send_msg(frag) ) {

                    pthread_mutex_lock(sendq_mutex);

                    --sendqcnt[frag->peer << 4];

                    next = frag->next;
                    if( prev ) {
                        prev->next = next;

                        if( !next ) {
                            list->tail = prev;
                        }
                    }
                    else {
                        list->head = next;
                    }

                    pthread_mutex_unlock(sendq_mutex);

                    freefrag(frag);
                }
                else {
                    skip[peer_node] = true;
                    skipped = true;
                    prev = frag;
                }
            }
            else {
                prev = frag;
            }

            frag = next;
        }
    }
    return NULL;
}
