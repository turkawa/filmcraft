//! Background Warp Stabilizer analysis. UI thread only schedules/reads atomics.
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use filmcraft_project::{ItemId, TrackItem, TrackMethod};
use filmcraft_time::Tick;
use crate::FilmcraftApp;

pub struct Job {
    sequence: ItemId,
    start: Tick,
    end: Tick,
    cancel: AtomicBool,
    phase: AtomicUsize,
    done: AtomicUsize,
    total: AtomicUsize,
    finished: AtomicBool,
    pub success: AtomicBool,
    acknowledged: AtomicBool,
}
impl Job {
    fn new(sequence: ItemId, item: &TrackItem) -> Self {
        Self { sequence, start: item.start, end: item.start + item.duration, cancel: AtomicBool::new(false), phase: AtomicUsize::new(1),
            done: AtomicUsize::new(0), total: AtomicUsize::new(0),
            finished: AtomicBool::new(false), success: AtomicBool::new(false),
            acknowledged: AtomicBool::new(false) }
    }
    pub fn status(&self) -> String {
        if self.finished.load(Ordering::Acquire) {
            return (if self.cancel.load(Ordering::Relaxed) { "Analysis cancelled" }
                else if self.success.load(Ordering::Relaxed) { "Analysis complete" }
                else { "Analysis failed" }).to_owned();
        }
        let phase = self.phase.load(Ordering::Relaxed);
        let done = self.done.load(Ordering::Relaxed);
        let total = self.total.load(Ordering::Relaxed);
        let pct = if total == 0 { 0 } else { done.saturating_mul(100) / total };
        format!("Analyzing: {pct}% (frame {done} of {total}) — step {phase} of 2")
    }
    pub fn phase(&self) -> usize { self.phase.load(Ordering::Relaxed) }
    pub fn running(&self) -> bool { !self.finished.load(Ordering::Acquire) }
    pub fn cancel(&self) { self.cancel.store(true, Ordering::Release); }
}
fn jobs() -> &'static Mutex<HashMap<u64, Arc<Job>>> {
    static JOBS: OnceLock<Mutex<HashMap<u64, Arc<Job>>>> = OnceLock::new();
    JOBS.get_or_init(Default::default)
}

pub fn poll_completed(app: &mut FilmcraftApp) {
    if let Ok(map) = jobs().lock() {
        for job in map.values() {
            if job.finished.load(Ordering::Acquire) && job.success.load(Ordering::Acquire)
                && !job.acknowledged.swap(true, Ordering::AcqRel) {
                app.stabilizer_revision = app.stabilizer_revision.wrapping_add(1);
            }
        }
    }
}

pub fn retry(item: &TrackItem, method: TrackMethod, detailed: bool) {
    let key = filmcraft_render::vfx::stabilizer_cache_key(
        filmcraft_render::vfx::stabilizer_item_key(item), method, detailed);
    if let Ok(mut map) = jobs().lock() {
        if map.get(&key).is_some_and(|j| !j.running()) { map.remove(&key); }
    }
}

/// Prevent seeking from requesting decoder work for a clip under analysis.
pub fn blocks_preview(sequence: ItemId, time: Tick) -> bool {
    jobs().lock().ok().is_some_and(|map| map.values().any(|j|
        j.running() && j.sequence == sequence && time >= j.start && time < j.end
    ))
}

pub fn active() -> Option<Arc<Job>> {
    jobs().lock().ok()?.values().find(|j| j.running()).cloned()
}

/// Called from Effect Controls. At most one analysis per cache key.
pub fn ensure(app: &mut FilmcraftApp, item: &TrackItem, method: TrackMethod, detailed: bool, ctx: &egui::Context) -> Option<Arc<Job>> {
    let clip_key = filmcraft_render::vfx::stabilizer_item_key(item);
    let key = filmcraft_render::vfx::stabilizer_cache_key(clip_key, method, detailed);
    poll_completed(app);
    if filmcraft_render::vfx::cached_stabilizer_path(clip_key, method, detailed).is_some() {
        return None;
    }
    let job = {
        let mut map = jobs().lock().ok()?;
        if map.get(&key).is_some_and(|j| j.finished.load(Ordering::Acquire)
            && j.success.load(Ordering::Acquire)) {
            map.remove(&key); // cache was cleared since the previous successful run
        }
        if let Some(j) = map.get(&key) { j.clone() } else {
            let seq_id = app.session.state.active_sequence?;
            let job = Arc::new(Job::new(seq_id, item));
            map.insert(key, job.clone());
            let project = app.session.project.clone();
            let seq_id = app.session.state.active_sequence?;
            let item = item.clone();
            let provider = app.session.media.provider(project.clone(), app.session.services.clone());
            let worker = job.clone();
            let repaint = ctx.clone();
            std::thread::Builder::new().name("filmcraft-stabilizer".into()).spawn(move || {
                let ok = project.sequence(seq_id).and_then(|seq| {
                    filmcraft_render::vfx::analyze_stabilizer_item(&project, seq, &item, &provider,
                        method, detailed, &worker.cancel, |phase, done, total| {
                            worker.phase.store(phase as usize, Ordering::Relaxed);
                            worker.done.store(done, Ordering::Relaxed);
                            worker.total.store(total, Ordering::Relaxed);
                            if done % 64 == 0 || done == total { repaint.request_repaint(); }
                        })
                }).is_some();
                worker.success.store(ok && !worker.cancel.load(Ordering::Acquire), Ordering::Release);
                worker.finished.store(true, Ordering::Release);
                repaint.request_repaint();
            }).ok()?;
            job
        }
    };
    Some(job)
}
