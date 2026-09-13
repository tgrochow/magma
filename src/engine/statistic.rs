use std::collections::VecDeque;
use std::time::Instant;

const SIZE: usize = 5;

pub struct Statistic {
    frame_count: i32,
    start: Instant,
    counts: VecDeque<i32>,
}

impl Statistic {
    pub fn new() -> Self {
        Statistic {
            frame_count: 0,
            start: Instant::now(),
            counts: VecDeque::with_capacity(SIZE),
        }
    }

    pub fn tick(&mut self) {
        self.frame_count += 1;
        if self.start.elapsed().as_secs() >= 1 {
            self.counts.push_back(self.frame_count);
            if self.counts.len() > SIZE {
                self.counts.pop_front();
            }
            self.start = Instant::now();
            self.frame_count = 0;
        }
    }

    pub fn calc_fps(&self) -> i32 {
        if self.counts.len() == 0 {
            return 0;
        }
        let mut sum: i32 = 0;
        for value in &self.counts {
            sum += value;
        }
        sum / self.counts.len() as i32
    }
}
