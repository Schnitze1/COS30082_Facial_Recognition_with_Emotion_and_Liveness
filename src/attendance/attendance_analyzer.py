import os
import csv
import sys
from datetime import datetime
from collections import defaultdict

try:
    from src.attendance.employee_manager import EmployeeManager
except ImportError:
    from employee_manager import EmployeeManager


class AttendanceAnalyzer:
    def __init__(self, attendance_dir: str = "src/attendance"):
        self.attendance_dir = attendance_dir
        self.log_file = os.path.join(self.attendance_dir, "attendance_log.csv")
        self.employee_manager = EmployeeManager(attendance_dir)
        self.grace_period_minutes = 5

    def parse_attendance_log(self, start_date: datetime = None, end_date: datetime = None) -> dict:
        # Read log file
        if not os.path.exists(self.log_file):
            return {}

        events_by_date = defaultdict(lambda: defaultdict(list))

        try:
            # Parse CSV rows
            with open(self.log_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        if not row.get("name"):
                            continue

                        timestamp_str = row.get("timestamp", "")
                        event = row.get("event", "")

                        # Only track ENTER/EXIT events
                        if not timestamp_str or event not in ["ENTER", "EXIT"]:
                            continue

                        event_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

                        # Filter by date range
                        if start_date and event_time < start_date:
                            continue
                        if end_date and event_time > end_date:
                            continue

                        event_date = event_time.date()
                        name = row.get("name", "").strip()

                        events_by_date[str(event_date)][name].append({
                            "event": event,
                            "time": event_time,
                            "timestamp_str": timestamp_str
                        })
                    except Exception as e:
                        pass

        except Exception as e:
            print(f"Error parsing attendance log: {e}")

        return dict(events_by_date)

    def analyze_day(self, date_str: str, events_by_name: dict) -> dict:
        # Parse date
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        analysis = {}

        # Get all employees
        employees = self.employee_manager.get_all_employees()

        # Analyze each employee
        for emp in employees:
            employee_id = emp.get("employee_id")
            name = emp.get("name")

            # Get expected hours
            expected_start, expected_end = self.employee_manager.get_expected_hours(employee_id, datetime.combine(date_obj, datetime.min.time()))

            status = "OFF" if expected_start == "OFF" else "ABSENT"
            actual_start = None
            deviation_minutes = None
            notes = ""

            # Check if employee logged in
            if expected_start != "OFF" and name in events_by_name:
                enter_events = [e for e in events_by_name[name] if e["event"] == "ENTER"]
                if enter_events:
                    enter_events.sort(key=lambda e: e["time"])
                    actual_start = enter_events[0]["time"]

                    # Compare against grace period
                    expected_dt = datetime.strptime(f"{date_str} {expected_start}", "%Y-%m-%d %H:%M")
                    grace_cutoff = expected_dt.timestamp() + (self.grace_period_minutes * 60)

                    if actual_start.timestamp() <= grace_cutoff:
                        status = "ON_TIME"
                        deviation_minutes = int((actual_start.timestamp() - expected_dt.timestamp()) / 60)
                    else:
                        status = "LATE"
                        deviation_minutes = int((actual_start.timestamp() - expected_dt.timestamp()) / 60)

            analysis[employee_id] = {
                "employee_id": employee_id,
                "name": name,
                "expected_start": expected_start,
                "actual_start": actual_start.strftime("%H:%M") if actual_start else "N/A",
                "status": status,
                "deviation_minutes": deviation_minutes if deviation_minutes is not None else "N/A",
                "notes": notes
            }

        # Track unknown faces
        unknown_names = set(events_by_name.keys()) - set(emp.get("name") for emp in employees)
        for unknown_name in unknown_names:
            enter_events = [e for e in events_by_name[unknown_name] if e["event"] == "ENTER"]
            if enter_events:
                enter_events.sort(key=lambda e: e["time"])
                actual_start = enter_events[0]["time"]

                analysis[f"UNKNOWN_{unknown_name}"] = {
                    "employee_id": "UNKNOWN",
                    "name": unknown_name,
                    "expected_start": "N/A",
                    "actual_start": actual_start.strftime("%H:%M"),
                    "status": "UNKNOWN",
                    "deviation_minutes": "N/A",
                    "notes": "Unrecognized face in attendance log"
                }

        return analysis

    def analyze_date_range(self, start_date: datetime, end_date: datetime) -> dict:
        events_by_date = self.parse_attendance_log(start_date, end_date)
        analysis_by_date = {}

        for date_str in sorted(events_by_date.keys()):
            events_by_name = events_by_date[date_str]
            analysis_by_date[date_str] = self.analyze_day(date_str, events_by_name)

        return analysis_by_date

    def get_attendance_summary_stats(self, analysis: dict) -> dict:
        counts = {
            "ON_TIME": 0,
            "LATE": 0,
            "ABSENT": 0,
            "OFF": 0,
            "UNKNOWN": 0,
            "TOTAL": 0
        }

        for record in analysis.values():
            status = record.get("status")
            if status in counts:
                counts[status] += 1
                if status != "OFF":
                    counts["TOTAL"] += 1

        return counts

    def flag_exceptions(self, analysis: dict, statuses: list = None) -> list:
        if statuses is None:
            statuses = ["LATE", "ABSENT", "UNKNOWN"]

        exceptions = []
        for record in analysis.values():
            if record.get("status") in statuses:
                exceptions.append(record)

        return sorted(exceptions, key=lambda x: x.get("status"))

    def detect_unknown_faces(self, start_date: datetime = None, end_date: datetime = None) -> list:
        events_by_date = self.parse_attendance_log(start_date, end_date)
        employees = self.employee_manager.get_all_employees()
        known_names = set(emp.get("name") for emp in employees)

        unknowns = []
        for date_str in sorted(events_by_date.keys()):
            for name, events in events_by_date[date_str].items():
                if name not in known_names:
                    enter_events = [e for e in events if e["event"] == "ENTER"]
                    if enter_events:
                        enter_events.sort(key=lambda e: e["time"])
                        first_entry = enter_events[0]["time"]
                        unknowns.append({
                            "date": date_str,
                            "name": name,
                            "first_entry": first_entry.strftime("%H:%M:%S"),
                            "event_count": len(events)
                        })

        return unknowns
