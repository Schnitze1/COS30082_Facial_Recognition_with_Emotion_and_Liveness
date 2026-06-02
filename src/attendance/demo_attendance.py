import os
import sys
import csv
from datetime import datetime, timedelta

sys.path.insert(0, 'src/attendance')

from employee_manager import EmployeeManager
from attendance_analyzer import AttendanceAnalyzer
from report_generator import ReportGenerator


def create_sample_attendance_log():
    # Start log
    log_file = "src/attendance/attendance_log.csv"

    # Reset for demo
    with open(log_file, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["name", "event", "spoken_statement", "emotion", "spoof_status", "timestamp"])

    today = datetime.now().date()
    test_events = [
        ("Person_001", "ENTER", today, "09:02"),
        ("Alice Johnson", "ENTER", today, "08:35"),
        ("Bob Smith", "ENTER", today, "10:10"),
        ("Charlie Brown", "ENTER", today, "09:00"),
        ("Diana Prince", "ENTER", today, "09:05"),
        ("Unknown_Person", "ENTER", today, "14:30"),
    ]

    # Write test events
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for name, event, date, time in test_events:
            timestamp = f"{date} {time}:00"
            writer.writerow([name, event, "", "", "", timestamp])

    print(f"[OK] Created sample attendance log for {today}")


def demo_workflow():
    print("\n" + "="*60)
    print("ATTENDANCE SYSTEM - DEMO")
    print("="*60)

    employees_sample = "src/attendance/employees_sample.csv"
    schedules_sample = "src/attendance/employee_schedules_sample.csv"

    if not os.path.exists(employees_sample) or not os.path.exists(schedules_sample):
        print("[ERROR] Sample data files not found. Please ensure employees_sample.csv and")
        print("  employee_schedules_sample.csv exist in src/attendance/")
        return

    manager = EmployeeManager()

    print("\n1. BULK IMPORTING EMPLOYEES")
    print("-" * 60)
    if manager.bulk_import_employees(employees_sample):
        print("[OK] Employee import successful")
    else:
        print("[FAILED] Employee import failed")

    print("\n2. BULK IMPORTING SCHEDULES")
    print("-" * 60)
    if manager.bulk_import_schedules(schedules_sample):
        print("[OK] Schedule import successful")
    else:
        print("[FAILED] Schedule import failed")

    manager.load_employees()
    manager.load_schedules()

    print("\n3. LOADED EMPLOYEES")
    print("-" * 60)
    for emp in manager.get_all_employees():
        print(f"  {emp['employee_id']}: {emp['name']} ({emp['department']})")

    print("\n4. CREATING SAMPLE ATTENDANCE LOG")
    print("-" * 60)
    create_sample_attendance_log()

    print("\n5. ANALYZING ATTENDANCE")
    print("-" * 60)
    analyzer = AttendanceAnalyzer()
    today = datetime.now()
    analysis = analyzer.analyze_date_range(today, today + timedelta(days=1))

    if analysis:
        today_str = today.strftime("%Y-%m-%d")
        if today_str in analysis:
            daily_analysis = analysis[today_str]
            stats = analyzer.get_attendance_summary_stats(daily_analysis)

            print(f"Attendance for {today_str}:")
            print(f"  On-time: {stats['ON_TIME']}")
            print(f"  Late: {stats['LATE']}")
            print(f"  Absent: {stats['ABSENT']}")
            print(f"  Off: {stats['OFF']}")
            print(f"  Unknown: {stats['UNKNOWN']}")

            print("\nDetailed Records:")
            for record in daily_analysis.values():
                status_mark = "[OK]" if record['status'] == "ON_TIME" else "[ISSUE]" if record['status'] in ["LATE", "ABSENT"] else "[?]"
                print(f"  {status_mark} {record['name']:20} {record['status']:10} (Expected: {record['expected_start']}, Actual: {record['actual_start']})")

            flagged = analyzer.flag_exceptions(daily_analysis, statuses=["LATE", "ABSENT", "UNKNOWN"])
            if flagged:
                print("\n[FLAGGED] Exceptions:")
                for exc in flagged:
                    print(f"  - {exc['name']}: {exc['status']}")

    print("\n6. GENERATING DAILY REPORT")
    print("-" * 60)
    generator = ReportGenerator()
    if generator.generate_daily_summary(today):
        print("[OK] Daily report generated")
        print(f"  Summary saved to: src/attendance/attendance_summary.csv")

        flagged_records = generator.generate_flagged_report(today)
        if flagged_records:
            print(f"[OK] Flagged report generated with {len(flagged_records)} exceptions")
    else:
        print("[FAILED] Daily report generation failed")

    print("\n7. ATTENDANCE STATISTICS")
    print("-" * 60)
    stats = generator.get_attendance_statistics(today)
    print(f"Date Range: {stats['DATE_RANGE']}")
    print(f"  On-time: {stats['ON_TIME']}")
    print(f"  Late: {stats['LATE']}")
    print(f"  Absent: {stats['ABSENT']}")
    print(f"  Off: {stats['OFF']}")
    print(f"  Unknown: {stats['UNKNOWN']}")

    print("\n" + "="*60)
    print("[OK] DEMO COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the generated CSV files in src/attendance/")
    print("2. Modify employees_sample.csv to link existing face database entries")
    print("3. Update employee_schedules_sample.csv with your actual schedules")
    print("4. Import employees and schedules using bulk_import_employees()")
    print("5. Run the attendance system and generate reports with generate_daily_summary()")


if __name__ == "__main__":
    try:
        demo_workflow()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
