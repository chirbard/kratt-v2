use super::with_write_lock;

#[test]
fn sync_operation_waits_for_write_pass() {
    let write_guard = super::acquire_write_lock();
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let (done_sender, done_receiver) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        let _ = started_sender.send(());
        with_write_lock(|| {
            let _ = done_sender.send(());
        });
    });
    assert!(started_receiver.recv().is_ok());
    assert!(
        done_receiver
            .recv_timeout(std::time::Duration::from_millis(50))
            .is_err()
    );
    drop(write_guard);
    assert!(
        done_receiver
            .recv_timeout(std::time::Duration::from_secs(1))
            .is_ok()
    );
    assert!(handle.join().is_ok());
}
