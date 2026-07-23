use std::time::{SystemTime, UNIX_EPOCH};

pub fn timestamp_now() -> String {
    let seconds_since_epoch = epoch_seconds_now();
    let days_since_epoch = (seconds_since_epoch / 86_400) as i64;
    let second_of_day = seconds_since_epoch % 86_400;
    let (year, month, day) = civil_from_days(days_since_epoch);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}",
        second_of_day / 3_600,
        (second_of_day % 3_600) / 60,
        second_of_day % 60
    )
}

pub fn today_iso() -> String {
    let seconds_since_epoch = epoch_seconds_now();
    let days_since_epoch = (seconds_since_epoch / 86_400) as i64;
    let (year, month, day) = civil_from_days(days_since_epoch);
    format!("{year:04}-{month:02}-{day:02}")
}

// The civil day before today, UTC — the daily report digests yesterday's
// ledger slice. Same epoch computation as today_iso, one day earlier.
pub fn yesterday_iso() -> String {
    let seconds_since_epoch = epoch_seconds_now().saturating_sub(86_400);
    let days_since_epoch = (seconds_since_epoch / 86_400) as i64;
    let (year, month, day) = civil_from_days(days_since_epoch);
    format!("{year:04}-{month:02}-{day:02}")
}

pub fn epoch_seconds_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub fn timestamp_to_epoch_seconds(timestamp: &str) -> Option<i64> {
    let (date, time) = timestamp.split_once('T')?;
    let mut date_parts = date.splitn(3, '-');
    let year: i64 = date_parts.next()?.parse().ok()?;
    let month: i64 = date_parts.next()?.parse().ok()?;
    let day: i64 = date_parts.next()?.parse().ok()?;
    let mut time_parts = time.splitn(3, ':');
    let hour: i64 = time_parts.next()?.parse().ok()?;
    let minute: i64 = time_parts.next()?.parse().ok()?;
    let second: i64 = time_parts.next()?.parse().ok()?;
    Some(days_from_civil(year, month, day) * 86_400 + hour * 3_600 + minute * 60 + second)
}

fn civil_from_days(days_since_epoch: i64) -> (i64, i64, i64) {
    let shifted_days = days_since_epoch + 719_468;
    let era = (if shifted_days >= 0 {
        shifted_days
    } else {
        shifted_days - 146_096
    }) / 146_097;
    let day_of_era = shifted_days - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year_base = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    };
    let year = if month <= 2 { year_base + 1 } else { year_base };
    (year, month, day)
}

fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let (adj_year, adj_month) = if month <= 2 {
        (year - 1, month + 9)
    } else {
        (year, month - 3)
    };
    let era = adj_year / 400;
    let year_of_era = adj_year - era * 400;
    let day_of_year = (153 * adj_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}
