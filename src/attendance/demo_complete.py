#!/usr/bin/env python3

import os
import sys
import csv
from datetime import datetime, timedelta

sys.path.insert(0, 'src/attendance')

from employee_manager import EmployeeManager
from attendance_analyzer import AttendanceAnalyzer
from report_generator import ReportGenerator


def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def demo():
    print_section("ATTENDANCE SYSTEM - DEMONSTRATION")

    attendance_dir = "src/attendance"
    manager = EmployeeManager(attendance_dir)
    analyzer = AttendanceAnalyzer(attendance_dir)
    generator = ReportGenerator(attendance_dir)

    print_section("1. IMPORTING SAMPLE EMPLOYEE DATA")

    employees_file = os.path.join(attendance_dir, "employees_extended.csv")
    schedules_file = os.path.join(attendance_dir, "employee_schedules_extended.csv")

    if os.path.exists(employees_file):
        print("Importing 5 sample employees linked to face database...")
        manager.bulk_import_employees(employees_file)
        print("Employee import complete")
    else:
        print("not found")

    if os.path.exists(schedules_file):
        print("\nImporting employee schedules...")
        manager.bulk_import_schedules(schedules_file)
        print("Schedule import complete")
    else:
        print("not found")

    print_section("2. EMPLOYEE ROSTER")

    manager.load_employees()
    employees = manager.get_all_employees()

    print(f"Total employees: {len(employees)}\n")
    print(f"{'ID':<6} {'Name':<20} {'Department':<15} {'Email':<30}")
    print("-" * 75)
    for emp in sorted(employees, key=lambda x: x['employee_id']):
        print(f"{emp['employee_id']:<6} {emp['name']:<20} {emp.get('department', 'N/A'):<15} {emp.get('email', 'N/A'):<30}")

    print_section("3. EMPLOYEE SCHEDULES")

    manager.load_schedules()
    schedules = manager.schedules

    print(f"Total schedules loaded: {len(schedules)}\n")
    for emp_id in sorted(list(schedules.keys())[:3]):
        sched = schedules[emp_id]
        emp = next((e for e in employees if e['employee_id'] == emp_id), {})
        print(f"{emp_id} - {emp.get('name', 'Unknown')}:")
        print(f"  Mon-Fri: {sched.get('monday_start')} - {sched.get('monday_end')}")
        print(f"  Sat:     {sched.get('saturday_start')} - {sched.get('saturday_end')}")
        print()

    print_section("4. LOADING SAMPLE ATTENDANCE DATA")

    sample_log = os.path.join(attendance_dir, "sample_attendance_log.csv")
    if os.path.exists(sample_log):
        from setup_demo import create_sample_attendance_log
        create_sample_attendance_log(attendance_dir)
        print(f"Generated attendance log with today's date")

        actual_log = os.path.join(attendance_dir, "attendance_log.csv")
        with open(actual_log, 'r') as f:
            count = sum(1 for _ in f) - 1
        print(f"  Total log entries: {count}")
    else:
        print("setup_demo.py not found")

    print_section("5. ATTENDANCE ANALYSIS")

    today = datetime.now()
    yesterday = today - timedelta(days=1)

    for target_date in [yesterday, today]:
        date_str = target_date.strftime("%Y-%m-%d")
        print(f"\n {date_str}:")
        print("-" * 70)

        start_dt = datetime.combine(target_date.date(), datetime.min.time())
        end_dt = start_dt + timedelta(days=1)

        analysis = analyzer.analyze_date_range(start_dt, end_dt)

        if not analysis or date_str not in analysis:
            print("  No attendance records")
            continue

        daily = analysis[date_str]
        stats = analyzer.get_attendance_summary_stats(daily)

        print(f"\n  Summary: On-time: {stats['ON_TIME']} | Late: {stats['LATE']} | Absent: {stats['ABSENT']} | Off: {stats['OFF']} | Unknown: {stats['UNKNOWN']}")

        print(f"\n  Records:")
        print(f"     {'Status':<12} {'Name':<20} {'Expected':<10} {'Actual':<10}")
        print("     " + "-" * 60)

        for record in sorted(daily.values(), key=lambda x: x['name']):
            status = record['status']
            icon = "✓" if status == "ON_TIME" else "✗" if status in ["LATE", "ABSENT"] else "?"
            print(f"     {icon} {status:<10} {record['name']:<20} {record['expected_start']:<10} {record['actual_start']:<10}")

        flagged = analyzer.flag_exceptions(daily, ["LATE", "ABSENT", "UNKNOWN"])
        if flagged:
            print(f"\n   Flagged ({len(flagged)} items):")
            for exc in flagged:
                dev = exc['deviation_minutes']
                print(f"     - {exc['name']:<20} {exc['status']:<10} (Deviation: {dev})")

    print_section("6. REPORT GENERATION")

    print("\nGenerating daily summary...")
    if generator.generate_daily_summary(today):
        print("Daily summary generated")
    else:
        print("Daily summary skipped")

    print("\nGenerating weekly summary...")
    week_start = today - timedelta(days=today.weekday())
    if generator.generate_weekly_summary(week_start, today):
        print("Weekly summary generated")
    else:
        print("Weekly summary skipped")

    print_section("7. FLAGGED RECORDS (Last 2 Days)")

    start_date = today - timedelta(days=2)
    flagged_records = generator.generate_flagged_report(start_date, today)

    if flagged_records:
        print(f"Found {len(flagged_records)} flagged records:\n")
        print(f"{'Date':<12} {'Employee':<20} {'Status':<10} {'Expected':<10} {'Actual':<10}")
        print("-" * 70)
        for r in flagged_records:
            print(f"{r['date']:<12} {r['name']:<20} {r['status']:<10} {r['expected_start']:<10} {r['actual_start']:<10}")
    else:
        print("No flagged records found!")

    print_section("8. ATTENDANCE STATISTICS (Last 7 Days)")

    week_ago = today - timedelta(days=7)
    stats = generator.get_attendance_statistics(week_ago, today)

    print(f"Period: {stats['DATE_RANGE']}")
    print(f"\nStats:")
    print(f"  ✓ On-time:   {stats['ON_TIME']:>3}")
    print(f"  ✗ Late:      {stats['LATE']:>3}")
    print(f"  ✗ Absent:    {stats['ABSENT']:>3}")
    print(f"  ⊘ Off:       {stats['OFF']:>3}")
    print(f"  ? Unknown:   {stats['UNKNOWN']:>3}")
    print(f"\n  Total Days: {stats['TOTAL_DAYS']}")

    if stats['TOTAL_DAYS'] > 0:
        total = stats['ON_TIME'] + stats['LATE'] + stats['ABSENT']
        on_time_rate = (stats['ON_TIME'] / total * 100) if total > 0 else 0
        print(f"  On-time Rate: {on_time_rate:.1f}%")

    print_section("9. EXPORT FUNCTIONALITY")

    export_file = os.path.join(attendance_dir, "export_sample.csv")
    if generator.export_report(week_ago, today, export_file, flagged_only=False):
        print(f"Full report exported: {export_file}")

    export_flagged = os.path.join(attendance_dir, "export_flagged_sample.csv")
    if generator.export_report(week_ago, today, export_flagged, flagged_only=True):
        print(f"Flagged report exported: {export_flagged}")

    print_section("10. UNKNOWN FACES DETECTION")

    unknowns = analyzer.detect_unknown_faces(week_ago, today)
    if unknowns:
        print(f"Detected {len(unknowns)} unknown persons:\n")
        print(f"{'Date':<12} {'Name':<20} {'First Entry':<10} {'Events':<8}")
        print("-" * 50)
        for u in unknowns:
            print(f"{u['date']:<12} {u['name']:<20} {u['first_entry']:<10} {u['event_count']:<8}")
    else:
        print("No unknown persons detected.")

    print_section("DEMONSTRATION COMPLETE")

    print("="*70)


if __name__ == "__main__":
    try:
        demo()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
