/*
 * Copyright (c) 2004-2011 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2009 The University of Tennessee and The University
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
 * Copyright (c) 2011-2014 NVIDIA Corporation.  All rights reserved.
 * Copyright (c) 2010-2012 IBM Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#include "ompi_config.h"
#include <errno.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif  /* HAVE_UNISTD_H */
#ifdef HAVE_STRING_H
#include <string.h>
#endif  /* HAVE_STRING_H */
#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif  /* HAVE_FCNTL_H */
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif  /* HAVE_SYS_TYPES_H */
#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif  /* HAVE_SYS_MMAN_H */
#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>  /* for mkfifo */
#endif  /* HAVE_SYS_STAT_H */

#include "ompi/constants.h"
#include "opal/mca/event/event.h"
#include "opal/util/bit_ops.h"
#include "opal/util/output.h"
#include "orte/util/proc_info.h"
#include "orte/util/show_help.h"
#include "orte/runtime/orte_globals.h"

#include "opal/mca/base/mca_base_param.h"
#include "ompi/mca/mpool/base/base.h"
#include "ompi/mca/btl/base/btl_base_error.h"
#include "ompi/mca/pml/ob1/pml_ob1_hdr.h"

#include "btl_nc.h"
 
 
static void fifo_push_back(fifolist_t* list, frag_t* frag);
static int mca_btl_nc_component_open(void);
static int mca_btl_nc_component_close(void);
static int nc_register(void);
static void processmsg(int type, const void* src, int size);
static void sbitreset(void* dst, const void* sbits, const void* src, int size8, int sbit);
static void memcpy8(void* to, const void* from, int n);
static void	copymsg(void* to, void* from, int sbit, int size8);
static mca_btl_base_module_t** mca_btl_nc_component_init(
    int *num_btls,
    bool enable_progress_threads,
    bool enable_mpi_threads
);

void memset8(void* to, uint64_t val, int n);
void sendack(int peer, void* hdr);
frag_t* allocfrag(int size);
void freefrag(frag_t* frag);


/*
 * Shared Memory (NC) component instance.
 */
mca_btl_nc_component_t mca_btl_nc_component = {
    {  /* super is being filled in */
        /* First, the mca_base_component_t struct containing meta information
          about the component itself */
        {
            MCA_BTL_BASE_VERSION_2_0_0,

            "nc", /* MCA component name */
            OMPI_MAJOR_VERSION,  /* MCA component major version */
            OMPI_MINOR_VERSION,  /* MCA component minor version */
            OMPI_RELEASE_VERSION,  /* MCA component release version */
            mca_btl_nc_component_open,  /* component open */
            mca_btl_nc_component_close,  /* component close */
            NULL,
            nc_register,
        },
        {
            /* The component is checkpoint ready */
            MCA_BASE_METADATA_PARAM_CHECKPOINT
        },

        mca_btl_nc_component_init,
        mca_btl_nc_component_progress,
    }  /* end super */
};


static int mca_btl_nc_component_verify(void) {
    return mca_btl_base_param_verify(&mca_btl_nc.super);
}


static int nc_register(void)
{
    mca_btl_nc.super.btl_exclusivity = MCA_BTL_EXCLUSIVITY_HIGH-1;
    mca_btl_nc.super.btl_eager_limit = MAX_EAGER_SIZE;
    mca_btl_nc.super.btl_rndv_eager_limit = MAX_EAGER_SIZE;
    mca_btl_nc.super.btl_max_send_size = MAX_MSG_SIZE;

    mca_btl_nc.super.btl_rdma_pipeline_send_length = MAX_MSG_SIZE;
    mca_btl_nc.super.btl_rdma_pipeline_frag_size = MAX_MSG_SIZE;
    mca_btl_nc.super.btl_min_rdma_pipeline_size = MAX_MSG_SIZE;

	mca_btl_nc.super.btl_flags = 0;

    mca_btl_nc.super.btl_seg_size = sizeof(mca_btl_nc_segment_t);
    mca_btl_nc.super.btl_bandwidth = 12000;  /* Mbs */
    mca_btl_nc.super.btl_latency   = 0.9;    /* Microsecs */

    /* Call the BTL based to register its MCA params */
    mca_btl_base_param_register(&mca_btl_nc_component.super.btl_version,
                                &mca_btl_nc.super);

    return mca_btl_nc_component_verify();
}

/*
 *  Called by MCA framework to open the component, registers
 *  component parameters.
 */

static int mca_btl_nc_component_open(void)
{
    return OMPI_SUCCESS;
}


/*
 * component cleanup - sanity checking of queue lengths
 */

static int mca_btl_nc_component_close(void)
{
	if( (uint64_t)mca_btl_nc_component.sendthread ) {

		mca_btl_nc_component.node->active = 0;
		__sfence();

		pthread_join(mca_btl_nc_component.sendthread, 0);
	}

    return OMPI_SUCCESS;
}


/*
 *  NCcomponent initialization
 */
mca_btl_base_module_t** mca_btl_nc_component_init(
    int *num_btls,
    bool enable_progress_threads,
    bool enable_mpi_threads)
{
    mca_btl_base_module_t **btls = NULL;

    *num_btls = 0;

    /* if no session directory was created, then we cannot be used */
    if (!orte_create_session_dirs) {
        return NULL;
    }
    
    /* allocate the Shared Memory BTL */
    *num_btls = 1;
    btls = (mca_btl_base_module_t**)malloc(sizeof(mca_btl_base_module_t*));
    if (NULL == btls) {
        return NULL;
    }

    /* get pointer to the btls */
    btls[0] = (mca_btl_base_module_t*)(&(mca_btl_nc));

    /* initialize some BTL data */
    /* start with no NC procs */
    mca_btl_nc_component.num_smp_procs = 0;
    mca_btl_nc_component.my_smp_rank   = -1;  /* not defined */
    /* set flag indicating btl not inited */
    mca_btl_nc.btl_inited = false;

    return btls;
}


static void fifo_push_back(fifolist_t* list, frag_t* frag)
{
	frag->next = 0;

	// transfer fragment to local nodes address space
	frag = (frag_t*)NODEADDR(frag);

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


static void processlist()
{
	fifolist_t* list = mca_btl_nc_component.myinq;

	assert( list->head );

	// transfer fragment from local nodes to peers address space
	frag_t* frag = list->head;
	frag = (frag_t*)((void*)frag + mca_btl_nc_component.shm_ofs);

	if( frag->next ) {
		list->head = frag->next;
	}
	else {
		__semlock(&list->lock);
		list->head = frag->next;
		__semunlock(&list->lock);
	}

	void* src = frag + 1;
	int size = frag->size;
	int type = frag->msgtype;

	processmsg(type, src, size);

	freefrag(frag);
}


int mca_btl_nc_component_progress(void)
{
	volatile fifolist_t* list = mca_btl_nc_component.myinq;

	if( list->head ) {
		processlist();
		return 1;
	}

	uint64_t shmofs = mca_btl_nc_component.shm_ofs;
	volatile ring_t* ring = mca_btl_nc_component.ring_desc;
	node_t* node = mca_btl_nc_component.node;
	int ring_cnt = node->ring_cnt;

	for( int ringndx = 0; ringndx < ring_cnt; ringndx++ ) {

		ring_t* ringnext = (ring_t*)(ring + 1);
		__builtin_prefetch((void*)ringnext);

		if( !__semtrylock(&ring->lock) ) {
			ring = ringnext;
			continue;
		}

		int sbit = ((ring->sbit) ^ 1);
		uint32_t rtail = ring->tail;
		rhdr_t* rhdr = (rhdr_t*)(ring->buf + shmofs + rtail);

		for( ; ; ) {

			int type = rhdr->type;

			if( (type & 1) != sbit ) {
				// no message
				int32_t z0 = ring->ttail;
				int32_t z1 = (rtail >> (RING_SIZE_LOG2 - 2));
				assert( z1 <= 3 ); // do 3 intermediate ring resets
				if( (z1 > z0) && __sync_bool_compare_and_swap(&ring->ttail, z0, z1) ) {
					// reset remote tail
					uint32_t* ptail = ring->ptail + (uint64_t)mca_btl_nc_component.sysctxt;
					__nccopy4(ptail, rtail);
				}
				break;
			}

			type &= ~1;
			int rsize = (rhdr->rsize << 2);

			if( type != MSG_TYPE_RST ) {

				assert( (rsize & 7) == 0 );
				assert( rsize >= 16 );

				bool local = (rhdr->dst_ndx == mca_btl_nc_component.cpuindex);
				bool done = false;
				frag_t* frag;

				int size8 = rsize - sizeof(rhdr_t);
				int ssize = rhdr->sync ? isyncsize(size8) : 0;
				if( size8 > SHDR ) {
					size8 -= ssize;
				}
				int size = size8 - rhdr->pad8;

				if( !(type & MSG_TYPE_BLK) ) {
					frag = allocfrag(rsize);
					if( !frag ) {
						return 0;
					}

					frag->msgtype = type;
					frag->size = size;

					if( rhdr->sync ) {
						void* src = (void*)(rhdr + 1) + ssize;
						void* sbits = (size8 <= SHDR) ? (void*)&rhdr->sbits : (void*)(rhdr + 1);
						sbitreset(frag + 1, sbits, src, size8, sbit);
					}
					else {
						copymsg(frag + 1, rhdr + 1, sbit, size8);
					}
					done = true;
				}
				else {
					frag = node->recvfrag[ringndx];

					if( !frag ) {
						int bufsize = sizeof(rhdr) + ((rhdr->sbits + 7) & ~7);

						frag = allocfrag(bufsize);
						if( !frag ) {
							return 0;
						}
						node->recvfrag[ringndx] = (void*)frag - shmofs;
						frag->size = 0;
					}
					else {
						frag = node->recvfrag[ringndx];
						frag = (frag_t*)((void*)frag + shmofs);
						assert( frag );
					}

					void* dst = (void*)(frag + 1) + frag->size;
					frag->size += size;

					if( rhdr->sync ) {
						void* src = (void*)(rhdr + 1) + ssize;
						void* sbits = (size8 <= SHDR) ? (void*)&rhdr->sbits : (void*)(rhdr + 1);
						sbitreset(dst, sbits, src, size8, sbit);
					}
					else {
						copymsg(dst, rhdr + 1, sbit, size8);
					}

					if( type == MSG_TYPE_BLKN ) {
						frag->msgtype = MSG_TYPE_FRAG;
						node->recvfrag[ringndx] = 0;
						done = true;
					}
				}

				rtail += rsize;		

				if( done ) {
					if( local ) {
						fifo_push_back((fifolist_t*)list, frag);
					}
					else {
						fifolist_t* list = mca_btl_nc_component.inq + rhdr->dst_ndx;
						fifo_push_back((fifolist_t*)list, frag);
					}
					break;
				}

				rhdr = (rhdr_t*)((uint8_t*)rhdr + rsize);			
			}
			else {
				// handle ring reset marker
				// clear rest of ring
				int n = RING_SIZE - rtail;
				if( n > 0 ) {
					memset8((uint8_t*)rhdr, sbit, n);
				}
				uint32_t* ptail = ring->ptail + (uint64_t)mca_btl_nc_component.sysctxt;
				__nccopy4(ptail, 0);

				rtail = 0;
				rhdr = (rhdr_t*)(ring->buf + shmofs);
				ring->ttail = 0;
				ring->sbit = sbit;
				sbit ^= 1;
			}		
		}

		ring->tail = rtail;
		__semunlock(&ring->lock);

		if( list->head ) {
			break;
		}

		ring = ringnext;
	}

	if( list->head ) {
		processlist();
		return 1;
	}
	return 0;
}


static void processmsg(int type, const void* src, int size)
{
	if( type == MSG_TYPE_ISEND ) {
		struct {
			mca_btl_base_descriptor_t base;
    		mca_btl_base_segment_t hdr_payload;
		} msg;

		msg.base.des_dst_cnt = 1;
	    msg.base.des_dst = &(msg.hdr_payload);

	    msg.hdr_payload.seg_addr.pval = (void*)src;
		msg.hdr_payload.seg_len = size; 

    	static mca_btl_active_message_callback_t* mreg =
				mca_btl_base_active_message_trigger + MCA_PML_OB1_HDR_TYPE_MATCH;

	    mreg->cbfunc(&mca_btl_nc.super, MCA_PML_OB1_HDR_TYPE_MATCH,	&(msg.base), mreg->cbdata);
	}
	else
	if( type == MSG_TYPE_FRAG ) {

        mca_btl_nc_hdr_t* hdr = (mca_btl_nc_hdr_t*)src;

        mca_btl_nc_hdr_t msg;

        msg.segment.base.seg_addr.pval = hdr + 1;
        msg.segment.base.seg_len = hdr->segment.base.seg_len;
        msg.base.des_dst_cnt = 1;
        msg.base.des_dst = &(msg.segment.base);

        mca_btl_active_message_callback_t* reg =
                        mca_btl_base_active_message_trigger + hdr->tag;

        reg->cbfunc(&mca_btl_nc.super, hdr->tag, &(msg.base), reg->cbdata);

        if( MCA_BTL_DES_SEND_ALWAYS_CALLBACK & hdr->base.des_flags ) {
			assert( hdr->self );
	        sendack(hdr->src_rank, hdr->self);
		}
	}
	else {
		assert( type == MSG_TYPE_ACK );

		frag_t* frag = *(frag_t**)src;
		mca_btl_nc_hdr_t* hdr = (mca_btl_nc_hdr_t*)(frag + 1);

        assert( MCA_BTL_DES_SEND_ALWAYS_CALLBACK & hdr->base.des_flags );

        hdr->base.des_cbfunc(&mca_btl_nc.super, hdr->endpoint, &hdr->base, OMPI_SUCCESS);
		
		freefrag(frag);
	}
}


static void sbitreset(void* dst, const void* sbits, const void* src, int size8, int sbit)
{
	// pointer to workload
	volatile uint64_t* p = (uint64_t*)src;
	assert( ((uint64_t)p & 0x7) == 0 );

	uint64_t* q = (uint64_t*)dst;
	assert( ((uint64_t)q & 0x7) == 0 );

	int n = (size8 >> 3);

	if( n <= 32 ) {
		uint32_t b = *(uint32_t*)sbits;
		b <<= (32 - n);

		for( ; ; ) {
			while( ((*p) & 1) != sbit ) {
				__lfence();
			}
			if( b & (1ul << 31) ) {
				(*q) = ((*p) | 1);
			}
			else {
				(*q) = ((*p) & ~1);
			}

			if( !--n ) {
				break;
			}
			++p;
			++q;
			b <<= 1;
		}
	}
	else {
		// pointer to sync bits
		volatile uint64_t* _sbits = (uint64_t*)sbits; 
		assert( ((uint64_t)_sbits & 0x7) == 0 );

		uint64_t b;
		int k = 0;

		for( ; ; ) {
			if( !k ) { 
				while( ((*_sbits) & 1) != sbit ) {
					__lfence();
				}
				b = *_sbits++;
				if( n >= 63 ) {
					k = 63;
				}
				else {
					k = n;
					b <<= (63 - k);
				}
			}
			while( ((*p) & 1) != sbit ) {
				__lfence();
			}
			if( b & (1ull << 63) ) {
				(*q) = ((*p) | 1);
			}
			else {
				(*q) = ((*p) & ~1);
			}

			if( !--n ) {
				break;
			}

			++p;
			++q;
			--k;
			b <<= 1;
		}
	}
}


static void memcpy8(void* to, const void* from, int n)
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
		"movq %%rax, (%%rdi)\n"
		"addq $8, %%rsi\n" 
		"addq $8, %%rdi\n" 
		"loop 1b\n"
	    : : "r" (n), "r" (from), "r" (to) : "ecx", "rax", "rsi", "rdi", "memory");
}


void memset8(void* to, uint64_t val, int n)
{
    assert( n > 0 );
	assert( (n & 0x7) == 0 );
	assert( (((uint64_t)to) & 0x7) == 0 );

	__asm__ __volatile__ (
		"movl %0, %%ecx\n"
		"shr  $3, %%ecx\n"
		"movq %1, %%rax\n"
		"movq %2, %%rdi\n"
		"1:\n"
		"movq %%rax, (%%rdi)\n"
		"addq $8, %%rdi\n" 
		"loop 1b\n"
	    : : "r" (n), "r" (val), "r" (to) : "ecx", "rax", "rdi", "memory");
}


static void	copymsg(void* to, void* from, int sbit, int size8)
{
    assert( size8 > 0 );
	assert( (size8 & 0x7) == 0 );
	assert( (((uint64_t)from) & 0x7) == 0 );
	assert( (((uint64_t)to) & 0x7) == 0 );

	uint64_t _sbit = sbit;

	__asm__ __volatile__ (
		"movl %0, %%ecx\n"
		"shr  $3, %%ecx\n"
		"movq %1, %%rbx\n"
		"movq %2, %%rsi\n"
		"movq %3, %%rdi\n"
		"1:\n"
		"movq (%%rsi), %%rax\n"
		"movq %%rax, (%%rdi)\n"
		"movq %%rbx, (%%rsi)\n"
		"addq $8, %%rsi\n" 
		"addq $8, %%rdi\n" 
		"loop 1b\n"
	    : : "r" (size8), "r" (_sbit), "r" (from), "r" (to) : "ecx", "rax", "rbx", "rsi", "rdi", "memory");
}

