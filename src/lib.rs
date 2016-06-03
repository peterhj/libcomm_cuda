extern crate array_cuda;

use array_cuda::device::{DeviceContext, DeviceBuffer};

use std::rc::{Rc};
use std::sync::{Arc, Barrier, Mutex};

pub struct DeviceBufRingCommBuilder {
  num_workers:  usize,
  barrier:  Arc<Barrier>,
  bufs:     Vec<Arc<Mutex<Option<Vec<DeviceBuffer<f32>>>>>>,
  //dst_bufs: Vec<Arc<Mutex<Option<Vec<DeviceBuffer<f32>>>>>>,
}

impl DeviceBufRingCommBuilder {
  pub fn new(num_workers: usize, buf_len: usize) -> DeviceBufRingCommBuilder {
    let pad = 32 * num_workers;
    let padded_len = (buf_len + pad - 1) / pad * pad;
    unimplemented!();
  }

  pub fn into_comm(self, worker_rank: usize, context: Rc<DeviceContext>) -> DeviceBufRingComm {
    DeviceBufRingComm{
      worker_rank:  worker_rank,
      num_workers:  self.num_workers,
      barrier:  self.barrier,
      context:  context,
      bufs:     self.bufs,
      //dst_bufs: self.dst_bufs,
    }
  }
}

pub struct DeviceBufRingComm {
  worker_rank:  usize,
  num_workers:  usize,
  barrier:  Arc<Barrier>,
  context:  Rc<DeviceContext>,
  bufs:     Vec<Arc<Mutex<Option<Vec<DeviceBuffer<f32>>>>>>,
  //dst_bufs: Vec<Arc<Mutex<Option<Vec<DeviceBuffer<f32>>>>>>,
}

impl DeviceBufRingComm {
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
      let mut dst = self.bufs[dst_rank].clone();
      let mut dst_guard = dst_parts.lock().unwrap();
      let mut dst_buf = &mut (*dst_guard).as_mut().unwrap()[part_idx];
      dst_buf.as_ref_mut(ctx).vector_sum(1.0, &src_buf.as_ref(ctx));
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
      let mut dst = self.bufs[dst_rank].clone();
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
