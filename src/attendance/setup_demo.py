#!/usr/bin/env python3

import os
import sys
import csv
import shutil
from datetime import datetime, timedelta

sys.path.insert(0, 'src/attendance')

from employee_manager import EmployeeManager


def reset_demo():
    attendance_dir = "src/attendance"

    print("\n" + "="*70)
    print("  ATTENDANCE SYSTEM - RESET & SETUP")
    print("="*70)

    print("\n1. CLEARING OLD DATA...")
    for f in ['attendance_log.csv', 'attendance_summary.csv', 'attendance_summary_flagged.csv',
              'employees.csv', 'employee_schedules.csv', 'export_sample.csv', 'export_flagged_sample.csv']:
        filepath = os.path.join(attendance_dir, f)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"    Cleared {f}")

    print("\n2. RE-INITIALIZING DATABASE...")
    manager = EmployeeManager(attendance_dir)
    manager._ensure_files_exist()
    print("    Database initialized")

    print("\n3. IMPORTING SAMPLE DATA...")
    employees_file = os.path.join(attendance_dir, "employees_sample.csv")
    schedules_file = os.path.join(attendance_dir, "employee_schedules_sample.csv")

    if os.path.exists(employees_file):
        manager.bulk_import_employees(employees_file)
        print("    Employees imported")

    if os.path.exists(schedules_file):
        manager.bulk_import_schedules(schedules_file)
        print("    Schedules imported")

    print("\n4. GENERATING SAMPLE ATTENDANCE LOG...")
    create_sample_attendance_log(attendance_dir)
    print("   Attendance log generated with today's date")

    print("\n" + "="*70)
    print("  SETUP COMPLETE - Ready for demo!")
    print("="*70 + "\n")


def create_sample_attendance_log(attendance_dir):
    log_file = os.path.join(attendance_dir, "attendance_log.csv")

    today = datetime.now()
    yesterday = today - timedelta(days=1)

    with open(log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "event", "spoken_statement", "timestamp"])

        events = [
            ("Person_001", "ENTER", ""),
            ("Person_001", "EXIT", ""),
            ("Person_002", "ENTER", ""),
            ("Person_002", "EXIT", ""),
            ("Person_003", "ENTER", ""),
            ("Person_003", "EXIT", ""),
            ("Person_004", "ENTER", ""),
            ("Person_004", "EXIT", ""),
            ("Unknown_Visitor", "ENTER", ""),
        ]

        for date_offset, date_obj in [(1, yesterday), (0, today)]:
            times = ["09:00", "17:00", "08:35", "16:30", "10:15", "18:00", "09:02", "17:00", "14:30"]
            for idx, (name, event, stmt) in enumerate(events):
                time = times[idx] if idx < len(times) else "12:00"
                timestamp = f"{date_obj.strftime('%Y-%m-%d')} {time}:00"
                writer.writerow([name, event, stmt, timestamp])


if __name__ == "__main__":
    reset_demo()
