import os
import csv
from datetime import datetime, timedelta

try:
    from src.attendance.attendance_analyzer import AttendanceAnalyzer
except ImportError:
    from attendance_analyzer import AttendanceAnalyzer


class ReportGenerator:
    def __init__(self, attendance_dir: str = "src/attendance"):
        self.attendance_dir = attendance_dir
        self.summary_file = os.path.join(self.attendance_dir, "attendance_summary.csv")
        self.flagged_file = os.path.join(self.attendance_dir, "attendance_summary_flagged.csv")
        self.analyzer = AttendanceAnalyzer(attendance_dir)
        self._ensure_summary_files_exist()

    def _ensure_summary_files_exist(self):
        # Create summary CSV
        if not os.path.exists(self.summary_file):
            with open(self.summary_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["date", "employee_id", "name", "expected_start", "actual_start", "status", "deviation_minutes", "notes"]
                )

        # Create flagged CSV
        if not os.path.exists(self.flagged_file):
            with open(self.flagged_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["date", "employee_id", "name", "expected_start", "actual_start", "status", "deviation_minutes", "notes"]
                )

    def generate_daily_summary(self, date: datetime) -> bool:
        # Format date range
        date_str = date.strftime("%Y-%m-%d")
        start_date = datetime.combine(date.date(), datetime.min.time())
        end_date = start_date + timedelta(days=1)

        # Analyze day
        analysis = self.analyzer.analyze_date_range(start_date, end_date)

        if not analysis or date_str not in analysis:
            print(f"No attendance data for {date_str}")
            return False

        daily_analysis = analysis[date_str]
        stats = self.analyzer.get_attendance_summary_stats(daily_analysis)

        # Prepare records
        records_to_write = []
        for record in daily_analysis.values():
            records_to_write.append([
                date_str,
                record.get("employee_id"),
                record.get("name"),
                record.get("expected_start"),
                record.get("actual_start"),
                record.get("status"),
                record.get("deviation_minutes"),
                record.get("notes")
            ])

        # Add summary row
        summary_row = [
            date_str,
            "SUMMARY",
            f"Total: {stats['ON_TIME']} on-time, {stats['LATE']} late, {stats['ABSENT']} absent, {stats['OFF']} off",
            "",
            "",
            "",
            "",
            f"Unknown: {stats['UNKNOWN']}"
        ]

        # Write to files
        self._append_to_summary_file(self.summary_file, records_to_write)
        self._append_to_summary_file(self.summary_file, [summary_row])

        # Write flagged records
        flagged_records = [r for r in records_to_write if r[5] in ["LATE", "ABSENT", "UNKNOWN"]]
        if flagged_records:
            self._append_to_summary_file(self.flagged_file, flagged_records)

        print(f"Daily summary generated for {date_str}")
        print(f"  On-time: {stats['ON_TIME']}, Late: {stats['LATE']}, Absent: {stats['ABSENT']}, Off: {stats['OFF']}, Unknown: {stats['UNKNOWN']}")

        return True

    def generate_weekly_summary(self, start_date: datetime, end_date: datetime = None) -> bool:
        if end_date is None:
            end_date = start_date + timedelta(days=6)

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        analysis = self.analyzer.analyze_date_range(start_date, end_date)

        if not analysis:
            print(f"No attendance data for week {start_str} to {end_str}")
            return False

        records_to_write = []
        total_stats = {
            "ON_TIME": 0,
            "LATE": 0,
            "ABSENT": 0,
            "OFF": 0,
            "UNKNOWN": 0,
            "TOTAL": 0
        }

        for date_str in sorted(analysis.keys()):
            daily_analysis = analysis[date_str]
            daily_stats = self.analyzer.get_attendance_summary_stats(daily_analysis)

            for k in total_stats:
                total_stats[k] += daily_stats[k]

            for record in daily_analysis.values():
                records_to_write.append([
                    date_str,
                    record.get("employee_id"),
                    record.get("name"),
                    record.get("expected_start"),
                    record.get("actual_start"),
                    record.get("status"),
                    record.get("deviation_minutes"),
                    record.get("notes")
                ])

        summary_row = [
            f"{start_str} to {end_str}",
            "WEEKLY_SUMMARY",
            f"Total: {total_stats['ON_TIME']} on-time, {total_stats['LATE']} late, {total_stats['ABSENT']} absent, {total_stats['OFF']} off",
            "",
            "",
            "",
            "",
            f"Unknown: {total_stats['UNKNOWN']}"
        ]

        self._append_to_summary_file(self.summary_file, records_to_write)
        self._append_to_summary_file(self.summary_file, [summary_row])

        flagged_records = [r for r in records_to_write if r[5] in ["LATE", "ABSENT", "UNKNOWN"]]
        if flagged_records:
            self._append_to_summary_file(self.flagged_file, flagged_records)

        print(f"Weekly summary generated for {start_str} to {end_str}")
        print(f"  On-time: {total_stats['ON_TIME']}, Late: {total_stats['LATE']}, Absent: {total_stats['ABSENT']}, Off: {total_stats['OFF']}, Unknown: {total_stats['UNKNOWN']}")

        return True

    def generate_flagged_report(self, start_date: datetime, end_date: datetime = None, statuses: list = None) -> list:
        if statuses is None:
            statuses = ["LATE", "ABSENT", "UNKNOWN"]

        if end_date is None:
            end_date = start_date + timedelta(days=1)

        analysis = self.analyzer.analyze_date_range(start_date, end_date)
        flagged_entries = []

        for date_str in sorted(analysis.keys()):
            daily_analysis = analysis[date_str]
            for record in daily_analysis.values():
                if record.get("status") in statuses:
                    flagged_entries.append({
                        "date": date_str,
                        "employee_id": record.get("employee_id"),
                        "name": record.get("name"),
                        "expected_start": record.get("expected_start"),
                        "actual_start": record.get("actual_start"),
                        "status": record.get("status"),
                        "deviation_minutes": record.get("deviation_minutes"),
                        "notes": record.get("notes")
                    })

        return flagged_entries

    def get_employee_attendance_history(self, employee_id: str, start_date: datetime, end_date: datetime) -> list:
        analysis = self.analyzer.analyze_date_range(start_date, end_date)
        history = []

        for date_str in sorted(analysis.keys()):
            daily_analysis = analysis[date_str]
            for record in daily_analysis.values():
                if record.get("employee_id") == employee_id:
                    history.append({
                        "date": date_str,
                        "expected_start": record.get("expected_start"),
                        "actual_start": record.get("actual_start"),
                        "status": record.get("status"),
                        "deviation_minutes": record.get("deviation_minutes")
                    })

        return history

    def get_attendance_statistics(self, start_date: datetime, end_date: datetime = None) -> dict:
        if end_date is None:
            end_date = start_date + timedelta(days=1)

        analysis = self.analyzer.analyze_date_range(start_date, end_date)
        all_stats = {
            "ON_TIME": 0,
            "LATE": 0,
            "ABSENT": 0,
            "OFF": 0,
            "UNKNOWN": 0,
            "TOTAL_DAYS": 0,
            "DATE_RANGE": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        }

        for date_str in analysis.keys():
            daily_analysis = analysis[date_str]
            daily_stats = self.analyzer.get_attendance_summary_stats(daily_analysis)
            for k in ["ON_TIME", "LATE", "ABSENT", "OFF", "UNKNOWN"]:
                all_stats[k] += daily_stats[k]
            all_stats["TOTAL_DAYS"] += 1

        return all_stats

    def export_report(self, start_date: datetime, end_date: datetime = None, output_file: str = None, flagged_only: bool = False) -> bool:
        if end_date is None:
            end_date = start_date + timedelta(days=1)

        if output_file is None:
            date_str = start_date.strftime("%Y-%m-%d")
            output_file = os.path.join(self.attendance_dir, f"report_{date_str}.csv")

        flagged_entries = self.generate_flagged_report(start_date, end_date)

        if flagged_only and not flagged_entries:
            print(f"No flagged entries found for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            return False

        try:
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["date", "employee_id", "name", "expected_start", "actual_start", "status", "deviation_minutes", "notes"])
                writer.writeheader()

                if flagged_only:
                    writer.writerows(flagged_entries)
                else:
                    analysis = self.analyzer.analyze_date_range(start_date, end_date)
                    for date_str in sorted(analysis.keys()):
                        daily_analysis = analysis[date_str]
                        for record in daily_analysis.values():
                            writer.writerow({
                                "date": date_str,
                                "employee_id": record.get("employee_id"),
                                "name": record.get("name"),
                                "expected_start": record.get("expected_start"),
                                "actual_start": record.get("actual_start"),
                                "status": record.get("status"),
                                "deviation_minutes": record.get("deviation_minutes"),
                                "notes": record.get("notes")
                            })

            print(f"Report exported to {output_file}")
            return True
        except Exception as e:
            print(f"Error exporting report: {e}")
            return False

    def _append_to_summary_file(self, file_path: str, rows: list) -> None:
        try:
            with open(file_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
        except Exception as e:
            print(f"Error appending to {file_path}: {e}")
