extern crate array_cuda;

use array_cuda::device::{DeviceContext, DeviceBuffer, DeviceBufferInitExt};
use array_cuda::device::linalg::{VectorExt};

use std::rc::{Rc};
use std::sync::{Arc, Barrier, Mutex};

pub struct RingDeviceBufCommBuilder<T> where T: Copy {
  num_workers:  usize,
  buf_len:      usize,
  barrier:  Arc<Barrier>,
  bufs:     Vec<Arc<Mutex<Option<Vec<DeviceBuffer<T>>>>>>,
}

impl<T> RingDeviceBufCommBuilder<T> where T: Copy {
  pub fn new(num_workers: usize, buf_len: usize) -> RingDeviceBufCommBuilder<T> {
    let mut bufs = Vec::with_capacity(num_workers);
    for p in 0 .. num_workers {
      bufs.push(Arc::new(Mutex::new(None)));
    }
    RingDeviceBufCommBuilder{
      num_workers:  num_workers,
      buf_len:      buf_len,
      barrier:  Arc::new(Barrier::new(num_workers)),
      bufs:     bufs,
    }
  }
}

impl RingDeviceBufCommBuilder<f32> {
  pub fn into_comm(self, worker_rank: usize, context: Rc<DeviceContext>) -> RingDeviceBufComm<f32> {
    let pad = 32 * self.num_workers;
    let padded_whole_len = (self.buf_len + pad - 1) / pad * pad;
    let padded_part_len = padded_whole_len / self.num_workers;
    {
      let ctx = &(*context).as_ref();
      let mut parts = Vec::with_capacity(self.num_workers);
      for p in 0 .. self.num_workers {
        parts.push(DeviceBuffer::zeros(padded_part_len, ctx));
      }
      let mut w_parts = self.bufs[worker_rank].clone();
      let mut w_guard = w_parts.lock().unwrap();
      *w_guard = Some(parts);
    }
    RingDeviceBufComm{
      worker_rank:  worker_rank,
      num_workers:  self.num_workers,
      barrier:  self.barrier,
      context:  context,
      bufs:     self.bufs,
    }
  }
}

pub struct RingDeviceBufComm<T> where T: Copy {
  worker_rank:  usize,
  num_workers:  usize,
  barrier:  Arc<Barrier>,
  context:  Rc<DeviceContext>,
  pub bufs: Vec<Arc<Mutex<Option<Vec<DeviceBuffer<T>>>>>>,
}

impl RingDeviceBufComm<f32> {
  pub fn reduce_scatter(&self) {
    if self.num_workers < 2 {
      return;
    }
    let ctx = &(*self.context).as_ref();
    let num_rounds = self.num_workers - 1;
    for round in 0 .. num_rounds {
      let part_idx = (self.worker_rank + round + self.num_workers - 1) % self.num_workers;
      let dst_rank = (self.worker_rank + 1) % self.num_workers;
      let mut src_parts = self.bufs[self.worker_rank].clone();
      let mut src_guard = src_parts.lock().unwrap();
      let mut src_buf = &mut (*src_guard).as_mut().unwrap()[part_idx];
      let mut dst_parts = self.bufs[dst_rank].clone();
      let mut dst_guard = dst_parts.lock().unwrap();
      let mut dst_buf = &mut (*dst_guard).as_mut().unwrap()[part_idx];
      dst_buf.as_ref_mut(ctx).vector_add(1.0, &src_buf.as_ref(ctx));
      ctx.blocking_sync();
      self.barrier.wait();
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
      let mut src_parts = self.bufs[self.worker_rank].clone();
      let mut src_guard = src_parts.lock().unwrap();
      let mut src_buf = &mut (*src_guard).as_mut().unwrap()[part_idx];
      let mut dst_parts = self.bufs[dst_rank].clone();
      let mut dst_guard = dst_parts.lock().unwrap();
      let mut dst_buf = &mut (*dst_guard).as_mut().unwrap()[part_idx];
      dst_buf.as_ref_mut(ctx).copy(&src_buf.as_ref(ctx));
      ctx.blocking_sync();
      self.barrier.wait();
    }
  }

  pub fn allreduce(&self) {
    self.reduce_scatter();
    self.allgather();
  }
}
