use std::future::Future;
use std::sync::{Mutex, MutexGuard, OnceLock};

use tokio::runtime::Runtime;

#[allow(clippy::expect_used)]
pub(super) fn runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    // OnceLock::get_or_init requires FnOnce() -> T — no way to propagate errors.
    RUNTIME.get_or_init(|| Runtime::new().expect("tokio runtime"))
}

pub(crate) fn block_on<F>(future: F) -> F::Output
where
    F: Future,
{
    runtime().block_on(future)
}

// Turso sync and checkpoint operations mutate database-wide state and can
// dissolve a transaction on another connection. Every write pass and every
// sync operation must serialize on this lock.
static WRITE_LOCK: Mutex<()> = Mutex::new(());

pub(crate) fn acquire_write_lock() -> MutexGuard<'static, ()> {
    match WRITE_LOCK.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

pub(super) fn with_write_lock<Output>(operation: impl FnOnce() -> Output) -> Output {
    let _write_guard = acquire_write_lock();
    operation()
}
