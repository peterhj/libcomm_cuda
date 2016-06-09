extern crate array_cuda;

use array_cuda::device::{DeviceContext, DeviceBuffer, DeviceBufferInitExt, SharedDeviceBuffer, RawDeviceBuffer};
use array_cuda::device::linalg::{VectorExt, AsyncBlasVectorExt};

use std::rc::{Rc};
use std::sync::{Arc, Barrier, Mutex};

#[derive(Clone)]
pub struct RingDeviceBufCommBuilder<T> where T: Copy {
  num_workers:  usize,
  //buf_len:      usize,
  buf_len:  Arc<Mutex<Option<usize>>>,
  barrier:  Arc<Barrier>,
  //bufs_q:   Arc<Mutex<Vec<Option<Vec<Arc<SharedDeviceBuffer<T>>>>>>>,
  bufs_q:   Arc<Mutex<Vec<Option<Vec<Arc<RawDeviceBuffer<T>>>>>>>,
}

impl<T> RingDeviceBufCommBuilder<T> where T: Copy {
  pub fn new(num_workers: usize/*, buf_len: usize*/) -> RingDeviceBufCommBuilder<T> {
    /*let mut bufs = Vec::with_capacity(num_workers);
    for p in 0 .. num_workers {
      bufs.push(Arc::new(Mutex::new(None)));
    }*/
    RingDeviceBufCommBuilder{
      num_workers:  num_workers,
      //buf_len:      buf_len,
      buf_len:  Arc::new(Mutex::new(None)),
      barrier:  Arc::new(Barrier::new(num_workers)),
      //bufs:     bufs,
      bufs_q:   Arc::new(Mutex::new(vec![None; num_workers])),
    }
  }

  pub fn buf_len(&self) -> Option<usize> {
    let buf_len = self.buf_len.lock().unwrap();
    *buf_len
  }

  pub fn set_buf_len(&self, new_buf_len: usize) {
    let mut buf_len = self.buf_len.lock().unwrap();
    *buf_len = Some(new_buf_len);
  }

  pub fn try_set_buf_len(&self, new_buf_len: usize) -> usize {
    let mut buf_len = self.buf_len.lock().unwrap();
    if buf_len.is_none() {
      *buf_len = Some(new_buf_len);
    }
    (*buf_len).unwrap()
  }
}

impl RingDeviceBufCommBuilder<f32> {
  pub fn into_comm(self, worker_rank: usize, context: Rc<DeviceContext>) -> RingDeviceBufComm<f32> {
    let pad = 32 * self.num_workers;
    let buf_len = (*self.buf_len.lock().unwrap()).unwrap();
    let padded_whole_len = (buf_len + pad - 1) / pad * pad;
    let padded_part_len = padded_whole_len / self.num_workers;
    {
      let ctx = &(*context).as_ref();
      let mut parts = Vec::with_capacity(self.num_workers);
      for p in 0 .. self.num_workers {
        //parts.push(Arc::new(unsafe { SharedDeviceBuffer::new(padded_part_len, ctx) }));
        let part = unsafe { RawDeviceBuffer::new(padded_part_len, ctx) };
        part.as_ref().async_vector_scale(0.0, ctx);
        parts.push(Arc::new(part));
      }
      /*let mut w_parts = self.bufs[worker_rank].clone();
      let mut w_guard = w_parts.lock().unwrap();
      *w_guard = Some(parts);*/
      let mut bufs_q = self.bufs_q.lock().unwrap();
      bufs_q[worker_rank] = Some(parts);
    }
    self.barrier.wait();
    let bufs = self.bufs_q.lock().unwrap().clone().into_iter()
      .map(|maybe_parts| maybe_parts.unwrap())
      .collect();
    RingDeviceBufComm{
      worker_rank:  worker_rank,
      num_workers:  self.num_workers,
      barrier:  self.barrier,
      context:  context,
      //bufs:     self.bufs,
      bufs:     bufs,
    }
  }
}

pub struct RingDeviceBufComm<T> where T: Copy {
  worker_rank:  usize,
  num_workers:  usize,
  barrier:  Arc<Barrier>,
  context:  Rc<DeviceContext>,
  //pub bufs: Vec<Vec<Arc<SharedDeviceBuffer<T>>>>,
  pub bufs: Vec<Vec<Arc<RawDeviceBuffer<T>>>>,
  //pub bufs: Vec<OpCursor<Vec<Arc<RawDeviceBuffer<T>>>>>,
}

impl RingDeviceBufComm<f32> {
  pub fn worker_rank(&self) -> usize {
    self.worker_rank
  }

  pub fn num_workers(&self) -> usize {
    self.num_workers
  }

  pub fn reduce_scatter(&self) {
    if self.num_workers < 2 {
      return;
    }
    let ctx = &(*self.context).as_ref();
    let num_rounds = self.num_workers - 1;
    for round in 0 .. num_rounds {
      let part_idx = (self.worker_rank - round + self.num_workers - 1) % self.num_workers;
      let dst_rank = (self.worker_rank + 1) % self.num_workers;
      (*self.bufs[dst_rank][part_idx]).as_ref().async_vector_add(1.0, &(*self.bufs[self.worker_rank][part_idx]).as_ref(), ctx);
      if round < num_rounds - 1 {
        ctx.blocking_sync();
        self.barrier.wait();
      }
    }
  }

  pub fn allgather(&self) {
    if self.num_workers < 2 {
      return;
    }
    let ctx = &(*self.context).as_ref();
    let num_rounds = self.num_workers - 1;
    for round in 0 .. num_rounds {
      let part_idx = self.worker_rank;
      let dst_rank = (self.worker_rank + round + 1) % self.num_workers;
      self.bufs[self.worker_rank][part_idx].raw_send(&*self.bufs[dst_rank][part_idx], ctx);
      if round < num_rounds - 1 {
        ctx.blocking_sync();
        self.barrier.wait();
      }
    }
  }

  pub fn average(&self) {
    if self.num_workers < 2 {
      return;
    }
    let ctx = &(*self.context).as_ref();
    for part_idx in 0 .. self.num_workers {
      (*self.bufs[self.worker_rank][part_idx]).as_ref().async_vector_scale(1.0 / self.num_workers as f32, ctx);
    }
  }

  pub fn barrier(&self) {
    let ctx = &(*self.context).as_ref();
    ctx.blocking_sync();
    self.barrier.wait();
  }

  pub fn allreduce_sum(&self) {
    self.reduce_scatter();
    self.barrier();
    self.allgather();
  }

  pub fn allreduce_average(&self) {
    self.allreduce_sum();
    self.average();
  }
}
