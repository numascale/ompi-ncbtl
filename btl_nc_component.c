/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
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
static void processmsg(frag_t* frag);
static void memset8(void* to, uint64_t val, int n);
static void read_msg(void* dst, ring_t* r, const rhdr_t* rhdr, int sbit);
static mca_btl_base_module_t** mca_btl_nc_component_init(
    int *num_btls,
    bool enable_progress_threads,
    bool enable_mpi_threads
);

void sendack(int peer, frag_t* sfrag);
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

    mca_btl_nc_component.async_send = 0;
    mca_base_component_var_register(&mca_btl_nc_component.super.btl_version,
                                    "send_thread", "Whether or not to enable asynchronously send threads",
                                    MCA_BASE_VAR_TYPE_UNSIGNED_INT, NULL, 0, 0,
                                    OPAL_INFO_LVL_9, MCA_BASE_VAR_SCOPE_READONLY,
                                    &mca_btl_nc_component.async_send);

    mca_btl_nc_component.shared_queues = -1;
    mca_base_component_var_register(&mca_btl_nc_component.super.btl_version,
                                    "shared_queues", "Whether or not to enable shared queues",
                                    MCA_BASE_VAR_TYPE_UNSIGNED_INT, NULL, 0, 0,
                                    OPAL_INFO_LVL_9, MCA_BASE_VAR_SCOPE_READONLY,
                                    &mca_btl_nc_component.shared_queues);

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

        mca_btl_nc_component.nodedesc->active = 0;
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

    frag_t* frag = list->head;
    frag = (frag_t*)PROCADDR(frag);

    if( frag->next ) {
        list->head = frag->next;
    }
    else {
        __semlock(&list->lock);
        list->head = frag->next;
        __semunlock(&list->lock);
    }

    processmsg(frag);
}


static int nn = 0;


int mca_btl_nc_component_progress(void)
{
    volatile fifolist_t* list = mca_btl_nc_component.myinq;

    if( list->head ) {
        processlist();
        return 1;
    }

    bool qshared = mca_btl_nc_component.shared_queues;
    volatile node_t* nodedesc = mca_btl_nc_component.nodedesc;
    volatile ring_t* ring = nodedesc->ring;
    int ring_cnt = (int)nodedesc->ring_cnt;
    void* ring_base = mca_btl_nc_component.shm_ringbase;

    for( int i = 0; i < ring_cnt; i++ ) {

//	__builtin_prefetch((void*)(ring + 1));

        int rndx = ring->ndx - 1;

        if( (rndx < 0) || (qshared && !__semtrylock(&ring->lock)) ) {
            ++ring;
            continue;
        }

	    ring_t* r = (ring_t*)ring;

        int sbit = ((r->sbit) ^ 1);
        uint32_t rtail = r->tail;
        volatile rhdr_t* rhdr = (rhdr_t*)(ring_base + (rndx << RING_SIZE_LOG2) + rtail);

        for( ; ; ) {

            int type = rhdr->type;

            if( (type & 1) != sbit ) {

                // no message
                int32_t z0 = r->ttail;
                int32_t z1 = (rtail >> (RING_SIZE_LOG2 - 2));
                assert( z1 <= 3 ); // do 3 intermediate ring resets
                if( z1 > z0 ) {
                    // reset remote tail
                    uint32_t* ptail = mca_btl_nc_component.peer_stail[rndx] + mca_btl_nc_component.group;
                    __nccopy4(ptail, rtail);
                    r->ttail = z1;
                }
                break;
            }

            assert( type );
            type &= ~1;
            int size = rhdr->size;
            int size8 = ((size + 7) & ~7);

            if( type != MSG_TYPE_RST ) {

                bool done = false;
                frag_t* frag;

                if( !(type & MSG_TYPE_BLK) ) {
                    frag = allocfrag(size8);
                    if( !frag ) {
                        break;
                    }

                    frag->msgtype = type;
                    frag->size = size;
					frag->send = size;

                    read_msg(frag + 1, r, (rhdr_t*)rhdr, sbit);
                    done = true;
                }
                else {
                    frag = nodedesc->recvfrag[rndx];
                    if( !frag ) {
						// frag is first chunk of a multi chunk message

						mca_btl_nc_hdr_t* hdr = (mca_btl_nc_hdr_t*)(rhdr + 1);

						volatile uint64_t* p = (uint64_t*)(rhdr + 1);
						int n = sizeof(mca_btl_nc_hdr_t);
						n = (((n + 7) & ~7) >> 3);
						for( int i = 0; i < n; i++ ) {
							while( ((*p) & 1) != sbit ) {
								__asm__ __volatile__ ("pause");
       						}
							++p;
						}

						int totSize = hdr->size;
						totSize &= ~1;

                        frag = allocfrag(totSize);
                        if( !frag ) {
                            return 0;
                        }

                        nodedesc->recvfrag[rndx] = (void*)NODEADDR(frag);
                        frag->size = 0;
						frag->send = totSize;
                    }
                    else {
                        frag = nodedesc->recvfrag[rndx];
                        assert( frag );
                        frag = (frag_t*)PROCADDR(frag);
                    }

                    void* dst = (void*)(frag + 1) + frag->size;
                    frag->size += size;
					assert( frag->size <= frag->send );

                    read_msg(dst, r, (rhdr_t*)rhdr, sbit);

                    if( frag->size == frag->send ) {
                        frag->msgtype = MSG_TYPE_FRAG;
                        nodedesc->recvfrag[rndx] = 0;
                        done = true;
                    }
                }
                int rsize = RHDR_SIZE + syncsize(size8) + size8;
                rtail += rsize;

                if( done ) {
                    if( rhdr->dst_ndx == mca_btl_nc_component.cpuindex ) {
                        fifo_push_back((fifolist_t*)list, frag);
                    }
                    else {
                        fifolist_t* list = mca_btl_nc_component.inq + rhdr->dst_ndx;
                        fifo_push_back((fifolist_t*)list, frag);
                    }
                    break;
                }
                rhdr = (rhdr_t*)((uint8_t*)rhdr + rsize);
                break;
            }
            else {
                // handle ring reset marker
                // clear rest of ring
                int n = RING_SIZE - rtail;
                if( n > 0 ) {
                    memset8((uint8_t*)rhdr, sbit, n);
                }
                // use one cache line per counter
                uint32_t* ptail = mca_btl_nc_component.peer_stail[rndx] + mca_btl_nc_component.group;
                __nccopy4(ptail, 0);

                rtail = 0;
                rhdr = (rhdr_t*)(ring_base + (rndx << RING_SIZE_LOG2));

                r->ttail = 0;
                r->sbit = sbit;
                sbit ^= 1;
            }
        }

        r->tail = rtail;
        __semunlock(&r->lock);

        if( list->head ) {
            break;
        }

        ++ring;
    }
    if( list->head ) {
        processlist();
        return 1;
    }
    if( !mca_btl_nc_component.async_send && mca_btl_nc_component.pending_sends->head ) {
		if( qshared ) {
			send_sync_sharedq();
		}
		else {
			send_sync_p2p();
		}
    }
    return 0;
}


static void processmsg(frag_t* frag)
{
    int type = frag->msgtype;

    if( type == MSG_TYPE_ISEND ) {

        struct {
            mca_btl_base_descriptor_t base;
            mca_btl_base_segment_t hdr_payload;
        } msg;

        msg.base.des_dst_cnt = 1;
        msg.base.des_dst = &(msg.hdr_payload);

        msg.hdr_payload.seg_addr.pval = (void*)(frag + 1);
        msg.hdr_payload.seg_len = frag->size;

        static mca_btl_active_message_callback_t* mreg =
            mca_btl_base_active_message_trigger + MCA_PML_OB1_HDR_TYPE_MATCH;

        mreg->cbfunc(&mca_btl_nc.super, MCA_PML_OB1_HDR_TYPE_MATCH, &(msg.base), mreg->cbdata);

        freefrag(frag);
    }
    else
    if( type == MSG_TYPE_FRAG ) {
        mca_btl_nc_hdr_t* hdr = (mca_btl_nc_hdr_t*)(frag + 1);

        mca_btl_nc_hdr_t msg;

        msg.segment.base.seg_addr.pval = hdr + 1;
        msg.segment.base.seg_len = hdr->segment.base.seg_len;
        msg.base.des_dst_cnt = 1;
        msg.base.des_dst = &(msg.segment.base);

        mca_btl_active_message_callback_t* reg =
            mca_btl_base_active_message_trigger + hdr->tag;

        reg->cbfunc(&mca_btl_nc.super, hdr->tag, &(msg.base), reg->cbdata);

        assert( hdr->frag );
        sendack(hdr->src_rank, hdr->frag);

        if( frag->send >= 0 ) {
            freefrag(frag);
        }
    }
    else {
        assert( type == MSG_TYPE_ACK );

		// sender fragment
        frag_t* sfrag = *(frag_t**)(frag + 1);

		mca_btl_nc_hdr_t* hdr = (mca_btl_nc_hdr_t*)(sfrag + 1);
		if( sfrag->send >= 0 ) {
			hdr = (mca_btl_nc_hdr_t*)((void*)hdr + RHDR_SIZE);
		}

        if( MCA_BTL_DES_SEND_ALWAYS_CALLBACK & hdr->base.des_flags ) {
            hdr->base.des_cbfunc(&mca_btl_nc.super, hdr->endpoint, &hdr->base, OMPI_SUCCESS);
        }

        freefrag(sfrag);
        freefrag(frag);
    }
}


static void rcopy(void* dst, volatile void* src, int size, int sbit)
{
	assert( ((uint64_t)dst & 0x3) == 0 );
	assert( ((uint64_t)src & 0x3) == 0 );
	assert( (size > 0) && ((size & 7) == 0) );
	assert( (sbit == 0) || (sbit == 1) );

	volatile uint64_t* p = (uint64_t*)src;
	uint64_t* q = (uint64_t*)dst;

	int n = (size >> 3);

	while( n ) {

		int k = n;
		if( k > 63 ) {
			k = 63;
		}

        for( int i = 0; i < k; i++ ) {
      		// wait until receive completed
        	while( ((*p) & 1) != sbit ) {
	        	__asm__ __volatile__ ("pause");
       		}
    		*q++ = *p++;
        }

		// wait until receive completed
      	while( ((*p) & 1) != sbit ) {
	    	__asm__ __volatile__ ("pause");
		}
    	uint64_t b = *p++;
        b >>= 1;

		uint64_t* q0 = q;

		// restore lsb bits from b
		for( int i = 0; i < k; i++ ) {
			uint64_t z = *--q0;
			z >>= 1;
			z <<= 1;
			z |= (b & 1);
            b >>= 1;
    		*q0 = z;
		}

        n -= k;
	}
}


static void read_msg(void* dst, ring_t* r, const rhdr_t* rhdr, int sbit)
{
	assert( ((uint64_t)rhdr & 0x7) == 0 );
	assert( ((uint64_t)dst & 0x3) == 0 );

    int size8 = ((rhdr->size + 7) & ~7);

	volatile uint64_t* p = (uint64_t*)(rhdr + 1);
	uint64_t* q = (uint64_t*)dst;

	int n = (size8 >> 3);

    if( n ) {

		int k = n;
		if( k > 32 ) {
			k = 32;
		}

		// restore lsb bits from b
		uint64_t b = rhdr->sbits;

		for( int i = 0; i < k; i++ ) {
      		// wait until receive completed
			while( ((*p) & 1) != sbit ) {
				__asm__ __volatile__ ("pause");
       		}
    		*q++ = *p++;
		}

		uint64_t* q0 = q;

		for( int i = 0; i < k; i++ ) {
			uint64_t z = *--q0;
			z >>= 1;
			z <<= 1;
			z |= (b & 1);
			b >>= 1;
    		*q0 = z;
		}

		size8 -= 256;

		if( size8 > 0 ) {
			rcopy(q, p, size8, sbit);
		}
	}
}


static void memset8(void* to, uint64_t val, int n)
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
        "movnti %%rax, (%%rdi)\n"
        "addq $8, %%rdi\n"
        "loop 1b\n"
        "sfence\n"
        : : "r" (n), "r" (val), "r" (to) : "ecx", "rax", "rdi", "memory");
}
