    use super::recover_replica_once;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn startup_recovery_moves_replica_and_retries_once() {
        let directory = recovery_test_dir("retry-success");
        assert!(std::fs::create_dir_all(&directory).is_ok());
        let replica = directory.join("kratt-synced.db");
        assert!(std::fs::write(&replica, b"dirty").is_ok());
        assert!(std::fs::write(directory.join("kratt-synced.db-wal"), b"wal").is_ok());
        let mut calls = 0usize;
        let result = recover_replica_once(&replica, "2026-07-16", |recovering| {
            calls += 1;
            if calls == 1 {
                assert!(!recovering);
                panic!("dirty replica");
            }
            assert!(recovering);
            Ok(())
        });
        assert!(result.is_ok());
        assert_eq!(calls, 2);
        assert!(!replica.exists());
        assert!(
            directory
                .join("kratt-synced.db.corrupt-2026-07-16")
                .exists()
        );
        assert!(
            directory
                .join("kratt-synced.db.corrupt-2026-07-16-wal")
                .exists()
        );
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn startup_recovery_stops_after_second_failure() {
        let directory = recovery_test_dir("retry-failure");
        assert!(std::fs::create_dir_all(&directory).is_ok());
        let replica = directory.join("kratt-synced.db");
        assert!(std::fs::write(&replica, b"dirty").is_ok());
        let mut calls = 0usize;
        let result: Result<(), Box<dyn std::error::Error + Send + Sync>> =
            recover_replica_once(&replica, "2026-07-16", |_recovering| {
                calls += 1;
                Err(format!("probe failure {calls}"))
            });
        assert!(result.is_err());
        assert_eq!(calls, 2);
        assert!(!replica.exists());
        let message = match result {
            Err(err) => err.to_string(),
            Ok(()) => String::new(),
        };
        assert!(message.contains("after one retry"));
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn startup_probe_success_does_not_touch_replica() {
        let directory = recovery_test_dir("no-recovery");
        assert!(std::fs::create_dir_all(&directory).is_ok());
        let replica = directory.join("kratt-synced.db");
        assert!(std::fs::write(&replica, b"healthy").is_ok());
        let mut calls = 0usize;
        let result = recover_replica_once(&replica, "2026-07-16", |recovering| {
            calls += 1;
            assert!(!recovering);
            Ok(())
        });
        assert!(result.is_ok());
        assert_eq!(calls, 1);
        assert!(replica.exists());
        assert!(
            !directory
                .join("kratt-synced.db.corrupt-2026-07-16")
                .exists()
        );
        let _ = std::fs::remove_dir_all(directory);
    }

    fn recovery_test_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("kratt-{label}-{}-{nonce}", std::process::id()))
    }
