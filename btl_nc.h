/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2006-2007 Voltaire. All rights reserved.
 * Copyright (c) 2009-2010 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2010-2013 Los Alamos National Security, LLC.  
 *                         All rights reserved. 
 * Copyright (c) 2010-2012 IBM Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
/**
 * @file
 */
#ifndef MCA_BTL_NC_H
#define MCA_BTL_NC_H

#include "ompi_config.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#ifdef HAVE_STDINT_H
#include <stdint.h>
#endif  /* HAVE_STDINT_H */
#ifdef HAVE_SCHED_H
#include <sched.h>
#endif  /* HAVE_SCHED_H */

#include "opal/util/bit_ops.h"
#include "ompi/mca/btl/btl.h"
#include "ompi/mca/common/sm/common_sm.h"

BEGIN_C_DECLS

#define MY_RANK  mca_btl_nc_component.my_smp_rank

#define MSG_TYPE_ISEND 0x02
#define MSG_TYPE_FRAG  0x04
#define MSG_TYPE_ACK   0x06
#define MSG_TYPE_BLKN  0x08
#define MSG_TYPE_BLK   0x18
#define MSG_TYPE_RST   0x20


#define NC_CLSIZE (64)    // cache line size

#define MAX_NUMA_NODES 256
#define MAX_GROUPS 256
#define MAX_PROCS 4096

#define MAX_EAGER_SIZE (4 * 1024)
#define MAX_SEND_SIZE  (16 * 1024)
#define MAX_MSG_SIZE   (16 * 1024 * 1024)
#define MAX_SIZE_FRAGS (1024 * 1024 * 1024)

#define FRAG_SIZE 48
#define RHDR_SIZE 8
#define SHDR      256

#define syncsize(s) ((s <= SHDR) ? 0 : ((((s >> 3) + 62) / 63) << 3))
#define isyncsize(s) ((s <= SHDR) ? 0 : (((s - 16) >> 9) + 1) << 3)

//#define INTRA_NODE_DIST_MAX 12

#define RING_SIZE (256 * 1024)
#define RING_SIZE_LOG2 (18)

#define SEMLOCK_UNLOCKED 0
#define SEMLOCK_LOCKED 1

#define ALIGN8 __attribute__ ((aligned (8)))
#define ALIGN64 __attribute__ ((aligned (64)))
#define forceinline inline __attribute__((always_inline))

#define NODEADDR(ptr) ((void*)ptr - mca_btl_nc_component.shm_ofs)
#define PROCADDR(ptr) ((void*)ptr + mca_btl_nc_component.shm_ofs)


static void forceinline __semlock(volatile int32_t* v)
{
	__asm__ __volatile__ (
		"movl $1, %%ebx\n"
		"1:\n"
		"xorl %%eax, %%eax\n"
        "lock cmpxchgl %%ebx, %0\n"
		"jnz 1b\n"
        : "+m" (*v) : : "eax", "ebx", "memory");
}


static bool forceinline __semtrylock(volatile int32_t* v)
{
	int rc;
	__asm__ __volatile__ (
		"movl $1, %%ebx\n"
		"xorl %%eax, %%eax\n"
        "lock cmpxchgl %%ebx, %0\n"
		"xorl $1, %%eax\n"
		"movl %%eax, %1\n"
        : "+m" (*v), "=r"(rc) : : "eax", "ebx", "memory");
	return rc;
}


static void forceinline __semunlock(volatile int32_t* v)
{
	__asm__ __volatile__ (
	   "movl $0, (%0)\n"
        : : "r" (v) : "memory");
}


static inline int32_t lockedAdd(volatile int32_t* v, int val)
{
	return (int32_t)__sync_add_and_fetch(v, val);
}


static void forceinline __lfence() {
    __asm__ __volatile__ ("lfence\n");
}


static void forceinline __sfence() {
    __asm__ __volatile__ ("sfence\n");
}


static void forceinline __mfence() {
    __asm__ __volatile__ ("mfence\n");
}


static void forceinline __nccopy4(void* to, const uint32_t head)
{
    __asm__ __volatile__ (
		"movl %0, %%eax\n"
		"movnti %%eax, (%1)\n"
		"sfence\n"
        :: "r" (head), "r" (to) : "eax", "memory");
}


static void forceinline __nccopy8(void* to, const void* from)
{
    assert( (((uint64_t)from) & 0x7) == 0 );
    assert( (((uint64_t)to) & 0x7) == 0 );

    __asm__ __volatile__ (
        "movq (%0), %%rax\n"
        "movnti %%rax, (%1)\n"
        :: "r" (from), "r" (to) : "rax", "memory");
}


/*
 * ring header
 */
typedef struct {
	uint32_t type    : 6;  // sync bit and message type
	uint32_t dst_ndx : 6;  // destination cpu index in group sharing ring
	uint32_t rsize   : 16; // size in ring
	uint32_t pad8    : 3;  // padding bytes used for 8 bytes alignment
	uint32_t unused  : 1;  // unused
	uint32_t sbits;		   // synchronization bits
} rhdr_t;


/*  An abstraction that represents a connection to a endpoint process.
 *  An instance of mca_ptl_base_endpoint_t is associated w/ each process
 *  and BTL pair at startup.
 */

struct mca_btl_base_endpoint_t {
    int my_smp_rank;    /**< My SMP process rank.  Used for accessing
                         *   SMP specfic data structures. */
    int peer_smp_rank;  /**< My peer's SMP process rank.  Used for accessing
                         *   SMP specfic data structures. */
};


struct mca_btl_nc_segment_t {
    mca_btl_base_segment_t base;
};
typedef struct mca_btl_nc_segment_t mca_btl_nc_segment_t;


struct fifolist; // forward declaration

/*
 * message fragment
 */
typedef struct frag {
	int32_t      fsize;		// frag size
	bool         inuse;     // fragment is in use
	int32_t      size;		// message body size including mpi headers
	int32_t      prevsize;	// previous frag size
	int32_t      peer;
	int32_t      node;	    // target node
	int32_t      ofs;		// offset of data already procesed
	int32_t      rsize;     // total size in ring
	bool         lastfrag;  // last frag in pool
	int32_t      msgtype;
	struct frag* next;		// next in list, next in pool is determined by size
} frag_t;


/*
 * pending sends fifo list
 */
typedef struct fifolist {
	volatile int32_t lock;
	frag_t* head;			
	frag_t* tail;
} fifolist_t;


struct mca_btl_nc_hdr_t { 
    mca_btl_base_descriptor_t       base;
    mca_btl_nc_segment_t            segment;
    struct mca_btl_base_endpoint_t* endpoint;
    mca_btl_base_tag_t              tag;
	int32_t							reserve;  
	int32_t				            src_rank;
	frag_t*							self;
};
typedef struct mca_btl_nc_hdr_t mca_btl_nc_hdr_t;


struct msgstats_t {
    int32_t  maxrank;
    int32_t  rank;
    uint64_t totmsgs;
    int32_t  active;
    uint64_t bytessend;
    uint64_t dist[32];     // message size ditribution
};


typedef struct ring {
	volatile int32_t lock;
	uint32_t tail;  // read tail
	int32_t  sbit;  // sync bit
	int32_t  ttail; // ring reset flag
} ring_t;


typedef struct ringdesc {
	ring_t*          ring;			
	void*  	         ringbuf;
	int32_t          ndx;
	struct ringdesc* prev;
	struct ringdesc* next;
} ringdesc_t;


typedef struct ringlist {
	int32_t     cnt;
	ringdesc_t* head;			
	ringdesc_t* tail;
} ringlist_t;


typedef struct {
	volatile int32_t lock;
	bool     commited;
	uint32_t head;
	int32_t  sbit;
} pring_t;


typedef struct {
	uint32_t         stail[MAX_GROUPS];
	volatile int32_t fraglock ALIGN8;
	int32_t*         sendqcnt ALIGN8;
	int32_t          ring_cnt ALIGN8;
	bool             inuse[MAX_GROUPS * 2];
	volatile bool    active;
	int32_t          cpuid;
	bool             yieldcpu;
	void*            shm_base;
	void*	         shm_frags;		// base of frags in shared mem
	frag_t*          recvfrag[MAX_GROUPS];
	// !must be last member
	int32_t          ndxmax ALIGN8;	// max peer index on local node
} node_t;


typedef struct {
	volatile int32_t lock ALIGN8;
	int      readycnt;
	int      max_nodes;
	int      node_count;
	int      rank_count;
	int      num_smp_procs;
	uint32_t map[MAX_PROCS]; // (group | cpuindex)
	uint32_t cpuid[MAX_PROCS];
	int32_t  group[MAX_NUMA_NODES];
	int32_t  numadist[MAX_NUMA_NODES * MAX_NUMA_NODES];
	int32_t  numanode[MAX_PROCS];
} sysctxt_t;


/**
 * Shared Memory (NC) BTL module.
 */
struct mca_btl_nc_component_t {
    mca_btl_base_component_2_0_0_t super;  /**< base BTL component */

	int32_t    group;
	int32_t    cpuindex;
	int        statistics;			// create statistics
	int        grp_numa_dist;		// size of group in terms of numa distance
	uint8_t*   shm_stat;			// statistics buffer
	fifolist_t* pending_sends;

	sysctxt_t* sysctxt;				// global shm mapping info
	pthread_t  sendthread;

	uint32_t** stail;				// pointers to local send tails
	uint32_t** peer_stail;			// pointers to peers send tails

	fifolist_t* inq;				// input queues
	fifolist_t* myinq;				// local input queue
	int32_t*    sendqcnt;

	uint32_t*  map;

	node_t**   peer_node;
	node_t*    nodedesc;

	uint64_t   shmsize;
	void*      shm_base;
	void*      shm_ringbase;
	void*      shm_fragbase;
	uint64_t   shm_ofs;
	uint64_t   ring_ofs;
	uint64_t   frag_ofs;

	ring_t*    ring;
	pring_t*   peer_ring;
	void**     peer_ring_buf;
	ringlist_t ringlist;

    int32_t    num_smp_procs;		// current number of smp procs on this host
    int32_t    my_smp_rank;			// My SMP process rank.  Used for accessing
};
typedef struct mca_btl_nc_component_t mca_btl_nc_component_t;
OMPI_MODULE_DECLSPEC extern mca_btl_nc_component_t mca_btl_nc_component;


/**
 * NC BTL Interface
 */
struct mca_btl_nc_t {
    mca_btl_base_module_t  super;       /**< base BTL interface */
    bool btl_inited;  /**< flag indicating if btl has been inited */
    mca_btl_base_module_error_cb_fn_t error_cb;
};
typedef struct mca_btl_nc_t mca_btl_nc_t;
OMPI_MODULE_DECLSPEC extern mca_btl_nc_t mca_btl_nc;


/**
 * shared memory component progress.
 */
extern int mca_btl_nc_component_progress(void);


/**
 * Register a callback function that is called on error..
 *
 * @param btl (IN)     BTL module
 * @return             Status indicating if cleanup was successful
 */

int mca_btl_nc_register_error_cb(
    struct mca_btl_base_module_t* btl,
    mca_btl_base_module_error_cb_fn_t cbfunc
);

/**
 * Cleanup any resources held by the BTL.
 *
 * @param btl  BTL instance.
 * @return     OMPI_SUCCESS or error status on failure.
 */

extern int mca_btl_nc_finalize(
    struct mca_btl_base_module_t* btl
);


/**
 * PML->BTL notification of change in the process list.
 * PML->BTL Notification that a receive fragment has been matched.
 * Called for message that is send from process with the virtual
 * address of the shared memory segment being different than that of
 * the receiver.
 *
 * @param btl (IN)
 * @param proc (IN)
 * @param peer (OUT)
 * @return     OMPI_SUCCESS or error status on failure.
 *
 */

extern int mca_btl_nc_add_procs(
    struct mca_btl_base_module_t* btl,
    size_t nprocs,
    struct ompi_proc_t **procs,
    struct mca_btl_base_endpoint_t** peers,
    struct opal_bitmap_t* reachability
);


/**
 * PML->BTL notification of change in the process list.
 *
 * @param btl (IN)     BTL instance
 * @param proc (IN)    Peer process
 * @param peer (IN)    Peer addressing information.
 * @return             Status indicating if cleanup was successful
 *
 */
extern int mca_btl_nc_del_procs(
    struct mca_btl_base_module_t* btl,
    size_t nprocs,
    struct ompi_proc_t **procs,
    struct mca_btl_base_endpoint_t **peers
);


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
    uint32_t flags
);

/**
 * Return a segment allocated by this BTL.
 *
 * @param btl (IN)      BTL module
 * @param segment (IN)  Allocated segment.
 */
extern int mca_btl_nc_free(
    struct mca_btl_base_module_t* btl,
    mca_btl_base_descriptor_t* segment
);


/**
 * Pack data
 *
 * @param btl (IN)      BTL module
 * @param peer (IN)     BTL peer addressing
 */
struct mca_btl_base_descriptor_t* mca_btl_nc_prepare_src(
    struct mca_btl_base_module_t* btl,
    struct mca_btl_base_endpoint_t* endpoint,
    mca_mpool_base_registration_t* registration,
    struct opal_convertor_t* convertor,
    uint8_t order,
    size_t reserve,
    size_t* size,
    uint32_t flags
);


/**
 * Initiate an inlined send to the peer or return a descriptor.
 *
 * @param btl (IN)      BTL module
 * @param peer (IN)     BTL peer addressing
 */
extern int mca_btl_nc_sendi( struct mca_btl_base_module_t* btl,
                             struct mca_btl_base_endpoint_t* endpoint,
                             struct opal_convertor_t* convertor,
                             void* header,
                             size_t header_size,
                             size_t payload_size,
                             uint8_t order,
                             uint32_t flags,
                             mca_btl_base_tag_t tag,
                             mca_btl_base_descriptor_t** descriptor );

/**
 * Initiate a send to the peer.
 *
 * @param btl (IN)      BTL module
 * @param peer (IN)     BTL peer addressing
 */
extern int mca_btl_nc_send(
    struct mca_btl_base_module_t* btl,
    struct mca_btl_base_endpoint_t* endpoint,
    struct mca_btl_base_descriptor_t* descriptor,
    mca_btl_base_tag_t tag
);


/**
 * Fault Tolerance Event Notification Function
 * @param state Checkpoint Stae
 * @return OMPI_SUCCESS or failure status
 */
int mca_btl_nc_ft_event(int state);


//#define MCA_BTL_SM_SIGNAL_PEER(peer)

END_C_DECLS

#endif

