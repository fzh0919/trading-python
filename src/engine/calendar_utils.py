"""US, UK & EU market calendars, holiday/early-close checks, and trading-day math."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from pydantic import BaseModel

LONDON_TZ = ZoneInfo("Europe/London")
NY_TZ = ZoneInfo("America/New_York")
BERLIN_TZ = ZoneInfo("Europe/Berlin")

# Verified NYSE fixed/statutory closures + algorithmic fallback for future years
US_HOLIDAYS: set[str] = {
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
    # 2028
    "2028-01-17", "2028-02-21", "2028-04-14", "2028-05-29",
    "2028-06-19", "2028-07-04", "2028-09-04", "2028-11-23", "2028-12-25",
}

# Optional overrides for one-off statutory / royal closures
UK_BANK_HOLIDAYS_EXTRA: set[str] = set()
EU_HOLIDAYS_EXTRA: set[str] = set()

_MARKET_ALIASES: dict[str, str] = {
    "US": "US",
    "NYSE": "US",
    "NASDAQ": "US",
    "UK": "UK",
    "LSE": "UK",
    "GB": "UK",
    "EU": "EU",
    "EUROPE": "EU",
    "XETRA": "XETRA",
    "DE": "XETRA",
    "FRANKFURT": "XETRA",
    "EURONEXT": "EURONEXT",
    "PA": "EURONEXT",
    "AS": "EURONEXT",
    "PARIS": "EURONEXT",
    "AMSTERDAM": "EURONEXT",
}

_MARKET_TIMEZONES: dict[str, ZoneInfo] = {
    "US": NY_TZ,
    "UK": LONDON_TZ,
    "EU": BERLIN_TZ,
    "XETRA": BERLIN_TZ,
    "EURONEXT": BERLIN_TZ,
}


class UsMarketSchedule(BaseModel):
    ny_date: str
    uk_date: str
    uk_weekday_label: str
    open_time_uk: str
    macro_time_uk: str
    mid_morning_time_uk: str
    fomc_time_uk: str
    close_time_uk: str
    is_dst_mismatch: bool
    is_us_trading_day: bool
    is_us_holiday: bool
    is_us_early_close: bool = False
    close_time_et: str = "16:00"
    phase: str  # PRE | OPEN | CLOSED | HOLIDAY | WEEKEND
    minutes_to_open: int
    minutes_since_open: int
    minutes_to_close: int


def normalize_market(market: str) -> str:
    """Normalize exchange/country identifiers to canonical US, UK, EU, XETRA, or EURONEXT."""
    key = market.strip().upper()
    if key in _MARKET_ALIASES:
        return _MARKET_ALIASES[key]
    raise ValueError(f"Unsupported market identifier: {market!r}")


def infer_market_from_symbol(symbol: str) -> str:
    """Infer canonical trading calendar market from a Yahoo Finance ticker symbol."""
    upper = symbol.strip().upper()
    if upper.endswith(".L"):
        return "UK"
    if upper.endswith(".DE"):
        return "XETRA"
    if upper.endswith((".PA", ".AS", ".BR", ".LS", ".MI")):
        return "EURONEXT"
    return "US"


def _coerce_date(value: date | datetime | str, tz: ZoneInfo | None = None) -> date:
    if isinstance(value, datetime):
        if tz is not None and value.tzinfo is not None:
            return value.astimezone(tz).date()
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value.strip()[:10])
    raise TypeError(f"Unsupported date value: {type(value).__name__}")


def _easter_sunday(year: int) -> date:
    """Anonymous Gregorian Easter algorithm for computing Good Friday / Easter Monday."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _observed(d: date) -> date:
    if d.weekday() == 5:  # Saturday -> Friday
        return d - timedelta(days=1)
    if d.weekday() == 6:  # Sunday -> Monday
        return d + timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        nxt = date(year + 1, 1, 1)
    else:
        nxt = date(year, month + 1, 1)
    last = nxt - timedelta(days=1)
    offset = (last.weekday() - weekday) % 7
    return last - timedelta(days=offset)


def compute_nyse_holidays_for_year(year: int) -> set[str]:
    """Compute standard NYSE full-day holidays for any year."""
    good_friday = _easter_sunday(year) - timedelta(days=2)
    candidates = [
        _observed(date(year, 1, 1)),                 # New Year's Day
        _nth_weekday(year, 1, 0, 3),                 # MLK Jr. Day (3rd Mon Jan)
        _nth_weekday(year, 2, 0, 3),                 # Washington's Birthday (3rd Mon Feb)
        good_friday,                                 # Good Friday
        _last_weekday(year, 5, 0),                   # Memorial Day (Last Mon May)
        _observed(date(year, 6, 19)),                # Juneteenth
        _observed(date(year, 7, 4)),                 # Independence Day
        _nth_weekday(year, 9, 0, 1),                 # Labor Day (1st Mon Sep)
        _nth_weekday(year, 11, 3, 4),                # Thanksgiving (4th Thu Nov)
        _observed(date(year, 12, 25)),               # Christmas
    ]
    return {d.isoformat() for d in candidates}


def _is_us_holiday_date(d: date) -> bool:
    d_str = d.isoformat()
    if d_str in US_HOLIDAYS:
        return True
    return d_str in compute_nyse_holidays_for_year(d.year)


def is_us_market_holiday(now_utc: datetime | None = None) -> bool:
    dt_ny = (now_utc or datetime.now(timezone.utc)).astimezone(NY_TZ)
    return _is_us_holiday_date(dt_ny.date())


def compute_uk_bank_holidays_for_year(year: int) -> set[str]:
    """Compute statutory England & Wales (LSE) bank holidays for a given year."""
    easter = _easter_sunday(year)
    nyd = date(year, 1, 1)
    if nyd.weekday() == 5:
        nyd = nyd + timedelta(days=2)
    elif nyd.weekday() == 6:
        nyd = nyd + timedelta(days=1)

    xmas = date(year, 12, 25)
    if xmas.weekday() == 4:  # Fri Dec 25 -> Boxing Day Sat becomes Mon Dec 28
        xmas_pair = [date(year, 12, 25), date(year, 12, 28)]
    elif xmas.weekday() == 5:  # Sat Dec 25, Sun Dec 26 -> Mon Dec 27, Tue Dec 28
        xmas_pair = [date(year, 12, 27), date(year, 12, 28)]
    elif xmas.weekday() == 6:  # Sun Dec 25, Mon Dec 26 -> Mon Dec 26, Tue Dec 27
        xmas_pair = [date(year, 12, 26), date(year, 12, 27)]
    else:
        xmas_pair = [date(year, 12, 25), date(year, 12, 26)]

    holidays = [
        nyd,
        easter - timedelta(days=2),      # Good Friday
        easter + timedelta(days=1),      # Easter Monday
        _nth_weekday(year, 5, 0, 1),     # Early May bank holiday (1st Mon May)
        _last_weekday(year, 5, 0),       # Spring bank holiday (Last Mon May)
        _last_weekday(year, 8, 0),       # Summer bank holiday (Last Mon Aug)
        *xmas_pair,
    ]
    return {d.isoformat() for d in holidays}


def _is_uk_holiday_date(d: date) -> bool:
    d_str = d.isoformat()
    if d_str in UK_BANK_HOLIDAYS_EXTRA:
        return True
    return d_str in compute_uk_bank_holidays_for_year(d.year)


def is_uk_bank_holiday(now_utc: datetime | None = None) -> bool:
    """Return True when Europe/London date is an England & Wales (LSE) bank holiday."""
    dt_uk = (now_utc or datetime.now(timezone.utc)).astimezone(LONDON_TZ)
    return _is_uk_holiday_date(dt_uk.date())


def compute_eu_holidays_for_year(year: int, venue: str = "EU") -> set[str]:
    """Compute statutory pan-European exchange closures (Xetra / Euronext) for a given year.

    Continental cash equities (Xetra .DE, Euronext Paris .PA / Amsterdam .AS) share
    TARGET2-aligned closures (Jan 1, Good Friday, Easter Monday, May 1, Dec 25, Dec 26)
    without UK/US-style weekday substitution. Xetra additionally closes all day on Dec 24 & Dec 31.
    """
    norm_venue = normalize_market(venue)
    easter = _easter_sunday(year)
    holidays = [
        date(year, 1, 1),                # New Year's Day
        easter - timedelta(days=2),      # Good Friday
        easter + timedelta(days=1),      # Easter Monday
        date(year, 5, 1),                # Labour Day
        date(year, 12, 25),              # Christmas Day
        date(year, 12, 26),              # Boxing Day / St. Stephen's Day
    ]
    if norm_venue == "XETRA":
        holidays.extend([date(year, 12, 24), date(year, 12, 31)])
    return {d.isoformat() for d in holidays}


def _is_eu_holiday_date(d: date, venue: str = "EU") -> bool:
    d_str = d.isoformat()
    if d_str in EU_HOLIDAYS_EXTRA:
        return True
    return d_str in compute_eu_holidays_for_year(d.year, venue=venue)


def is_eu_market_holiday(now_utc: datetime | None = None, venue: str = "EU") -> bool:
    """Return True when Europe/Berlin date is a full-day closure for Xetra/Euronext."""
    dt_eu = (now_utc or datetime.now(timezone.utc)).astimezone(BERLIN_TZ)
    return _is_eu_holiday_date(dt_eu.date(), venue=venue)


def _is_us_early_close_date(d: date) -> bool:
    """NYSE 13:00 ET early close: Black Friday, Dec 24 (Mon-Thu), Jul 3 (Mon-Thu)."""
    if d.weekday() >= 5 or _is_us_holiday_date(d):
        return False
    black_friday = _nth_weekday(d.year, 11, 3, 4) + timedelta(days=1)
    if d == black_friday:
        return True
    # Christmas Eve & pre-Independence Day are 13:00 ET early closes only when Mon-Thu;
    # if they fall on Friday they are already full-day observed holidays.
    if d.weekday() < 4 and ((d.month == 12 and d.day == 24) or (d.month == 7 and d.day == 3)):
        return True
    return False


def _is_uk_early_close_date(d: date) -> bool:
    """LSE 12:30 UK early close: Christmas Eve & New Year's Eve when trading weekdays."""
    if d.weekday() >= 5 or _is_uk_holiday_date(d):
        return False
    return d.month == 12 and d.day in (24, 31)


def _is_eu_early_close_date(d: date, venue: str = "EURONEXT") -> bool:
    """Euronext 14:05 CET early close on Dec 24 & Dec 31 (Xetra is closed all day)."""
    norm_venue = normalize_market(venue)
    if norm_venue == "XETRA":
        return False
    if d.weekday() >= 5 or _is_eu_holiday_date(d, venue=norm_venue):
        return False
    return d.month == 12 and d.day in (24, 31)


def is_us_early_close(now_utc: datetime | None = None) -> bool:
    """Return True when NYSE has a scheduled 13:00 ET early close."""
    dt_ny = (now_utc or datetime.now(timezone.utc)).astimezone(NY_TZ)
    return _is_us_early_close_date(dt_ny.date())


def is_uk_early_close(now_utc: datetime | None = None) -> bool:
    """Return True when LSE has a scheduled 12:30 UK early close."""
    dt_uk = (now_utc or datetime.now(timezone.utc)).astimezone(LONDON_TZ)
    return _is_uk_early_close_date(dt_uk.date())


def is_eu_early_close(now_utc: datetime | None = None, venue: str = "EURONEXT") -> bool:
    """Return True when Euronext has a scheduled 14:05 CET early close."""
    dt_eu = (now_utc or datetime.now(timezone.utc)).astimezone(BERLIN_TZ)
    return _is_eu_early_close_date(dt_eu.date(), venue=venue)


def is_trading_day(day: date | datetime | str, market: str = "US") -> bool:
    """Return True if the given calendar date is an active trading session in `market`."""
    norm = normalize_market(market)
    d = _coerce_date(day, tz=_MARKET_TIMEZONES[norm])
    if d.weekday() >= 5:
        return False
    if norm == "US":
        return not _is_us_holiday_date(d)
    if norm == "UK":
        return not _is_uk_holiday_date(d)
    return not _is_eu_holiday_date(d, venue=norm)


def next_trading_day(
    start_date: date | datetime | str,
    market: str = "US",
    inclusive: bool = False,
) -> date:
    """Return the first trading day on/after `start_date` (strictly after if inclusive=False)."""
    norm = normalize_market(market)
    cur = _coerce_date(start_date, tz=_MARKET_TIMEZONES[norm])
    if not inclusive:
        cur = cur + timedelta(days=1)
    while not is_trading_day(cur, norm):
        cur = cur + timedelta(days=1)
    return cur


def add_trading_days(
    start_date: date | datetime | str,
    n_days: int,
    market: str = "US",
) -> date:
    """Advance `n_days` trading sessions from `start_date` in the specified market.

    Positive `n_days` counts full trading sessions strictly after `start_date`
    (e.g. `n_days=5` for T+5 settlement, skipping weekends and exchange holidays).
    Negative `n_days` walks backward by trading sessions; `0` returns `start_date` as a `date`.
    """
    norm = normalize_market(market)
    cur = _coerce_date(start_date, tz=_MARKET_TIMEZONES[norm])
    if n_days == 0:
        return cur
    step = 1 if n_days > 0 else -1
    remaining = abs(n_days)
    while remaining > 0:
        cur = cur + timedelta(days=step)
        if is_trading_day(cur, norm):
            remaining -= 1
    return cur


def get_market_close_utc(day: date | datetime | str, market: str = "US") -> datetime | None:
    """Return the scheduled closing instant in UTC for `day` in `market`, or None if closed."""
    norm = normalize_market(market)
    tz = _MARKET_TIMEZONES[norm]
    d = _coerce_date(day, tz=tz)
    if not is_trading_day(d, norm):
        return None
    if norm == "US":
        close_t = time(13, 0) if _is_us_early_close_date(d) else time(16, 0)
    elif norm == "UK":
        close_t = time(12, 30) if _is_uk_early_close_date(d) else time(16, 30)
    elif norm == "XETRA":
        close_t = time(17, 30)
    else:  # EU / EURONEXT
        close_t = time(14, 5) if _is_eu_early_close_date(d, venue=norm) else time(17, 30)
    return datetime.combine(d, close_t, tzinfo=tz).astimezone(timezone.utc)


def has_market_closed_on_date(
    target_date: date | datetime | str,
    market: str = "US",
    now_utc: datetime | None = None,
) -> bool:
    """Return True once the target date's session has officially closed (safe for T+N settlement)."""
    norm = normalize_market(market)
    tz = _MARKET_TIMEZONES[norm]
    d = _coerce_date(target_date, tz=tz)
    base_utc = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    close_utc = get_market_close_utc(d, norm)
    if close_utc is not None:
        return base_utc >= close_utc
    return base_utc.astimezone(tz).date() > d


def get_market_calendar_alerts(now_utc: datetime | None = None) -> list[str]:
    """Return concise UK/EU/US holiday or early-close notices for optional prompt context."""
    base = now_utc or datetime.now(timezone.utc)
    dt_uk = base.astimezone(LONDON_TZ)
    dt_eu = base.astimezone(BERLIN_TZ)
    dt_ny = base.astimezone(NY_TZ)
    alerts: list[str] = []

    if _is_uk_holiday_date(dt_uk.date()):
        alerts.append("🇬🇧 今日英国 Bank Holiday（LSE 休市）")
    elif _is_uk_early_close_date(dt_uk.date()):
        alerts.append("🇬🇧 今日 LSE 半日市（12:30 UK 提前收盘）")

    if _is_eu_holiday_date(dt_eu.date(), venue="EU"):
        alerts.append("🇪🇺 今日欧洲主要交易所（Xetra/Euronext）休市")
    elif _is_eu_holiday_date(dt_eu.date(), venue="XETRA"):
        alerts.append("🇪🇺 今日 Xetra 休市 / Euronext 半日市（14:05 CET 收盘）")

    if _is_us_holiday_date(dt_ny.date()):
        alerts.append("🇺🇸 今日美股法定休市")
    elif _is_us_early_close_date(dt_ny.date()):
        sched = get_us_market_schedule_uk(base)
        alerts.append(f"🇺🇸 今日美股半日市（13:00 ET / {sched.close_time_uk} UK 提前收盘）")

    return alerts


def get_uk_weekday_label(now_utc: datetime | None = None) -> str:
    dt_uk = (now_utc or datetime.now(timezone.utc)).astimezone(LONDON_TZ)
    labels = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return labels[dt_uk.weekday()]


def is_london_weekday(now_utc: datetime | None = None) -> bool:
    dt_uk = (now_utc or datetime.now(timezone.utc)).astimezone(LONDON_TZ)
    return dt_uk.weekday() < 5


def get_us_market_schedule_uk(now_utc: datetime | None = None) -> UsMarketSchedule:
    base = now_utc or datetime.now(timezone.utc)
    dt_uk = base.astimezone(LONDON_TZ)
    dt_ny = base.astimezone(NY_TZ)

    # Actual UTC offset difference handles DST mismatch weeks cleanly
    uk_offset_h = int((dt_uk.utcoffset() or timedelta(0)).total_seconds() // 3600)
    ny_offset_h = int((dt_ny.utcoffset() or timedelta(0)).total_seconds() // 3600)
    diff_hours = uk_offset_h - ny_offset_h  # Normally 5, during US/UK DST gap = 4

    def _to_uk(et_h: int, et_m: str) -> str:
        return f"{et_h + diff_hours:02d}:{et_m}"

    ny_weekday = dt_ny.weekday()
    holiday = _is_us_holiday_date(dt_ny.date())
    trading_day = (ny_weekday < 5) and (not holiday)
    early_close = trading_day and _is_us_early_close_date(dt_ny.date())

    ny_min = dt_ny.hour * 60 + dt_ny.minute
    open_min = 9 * 60 + 30
    close_hour = 13 if early_close else 16
    close_min = close_hour * 60

    if ny_weekday >= 5:
        phase = "WEEKEND"
    elif holiday:
        phase = "HOLIDAY"
    elif ny_min >= close_min:
        phase = "CLOSED"
    elif ny_min >= open_min:
        phase = "OPEN"
    else:
        phase = "PRE"

    return UsMarketSchedule(
        ny_date=dt_ny.date().isoformat(),
        uk_date=dt_uk.date().isoformat(),
        uk_weekday_label=get_uk_weekday_label(base),
        open_time_uk=_to_uk(9, "30"),
        macro_time_uk=_to_uk(8, "30"),
        mid_morning_time_uk=_to_uk(10, "00"),
        fomc_time_uk=_to_uk(14, "00"),
        close_time_uk=_to_uk(close_hour, "00"),
        is_dst_mismatch=(diff_hours != 5),
        is_us_trading_day=trading_day,
        is_us_holiday=holiday,
        is_us_early_close=early_close,
        close_time_et=f"{close_hour:02d}:00",
        phase=phase,
        minutes_to_open=open_min - ny_min,
        minutes_since_open=ny_min - open_min,
        minutes_to_close=close_min - ny_min,
    )
