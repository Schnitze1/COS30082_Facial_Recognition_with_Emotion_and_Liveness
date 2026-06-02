import os
import csv
from datetime import datetime
from pathlib import Path


class EmployeeManager:
    def __init__(self, attendance_dir: str = "src/attendance"):
        self.attendance_dir = attendance_dir
        self.db_path = os.path.join(self.attendance_dir, "faces_db")
        self.employees_file = os.path.join(self.attendance_dir, "employees.csv")
        self.schedules_file = os.path.join(self.attendance_dir, "employee_schedules.csv")
        self.employees = {}
        self.schedules = {}

    def _ensure_files_exist(self):
        # Initialize directories
        os.makedirs(self.attendance_dir, exist_ok=True)
        os.makedirs(self.db_path, exist_ok=True)

        # Create employees CSV
        if not os.path.exists(self.employees_file):
            with open(self.employees_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["employee_id", "name", "person_dir", "department", "email"])

        # Create schedules CSV
        if not os.path.exists(self.schedules_file):
            with open(self.schedules_file, "w", newline="", encoding="utf-8") as f:
                headers = ["employee_id", "monday_start", "monday_end", "tuesday_start", "tuesday_end",
                          "wednesday_start", "wednesday_end", "thursday_start", "thursday_end",
                          "friday_start", "friday_end", "saturday_start", "saturday_end",
                          "sunday_start", "sunday_end", "notes"]
                csv.writer(f).writerow(headers)

    def load_employees(self) -> dict:
        self._ensure_files_exist()
        self.employees = {}
        try:
            with open(self.employees_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["employee_id"] and row["name"]:
                        self.employees[row["name"]] = row
        except Exception as e:
            print(f"Error loading employees: {e}")
        return self.employees

    def load_schedules(self) -> dict:
        self._ensure_files_exist()
        self.schedules = {}
        try:
            with open(self.schedules_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["employee_id"]:
                        self.schedules[row["employee_id"]] = row
        except Exception as e:
            print(f"Error loading schedules: {e}")
        return self.schedules

    def get_employee_by_name(self, name: str) -> dict:
        if not self.employees:
            self.load_employees()
        return self.employees.get(name)

    def get_schedule_by_employee_id(self, employee_id: str) -> dict:
        if not self.schedules:
            self.load_schedules()
        return self.schedules.get(employee_id)

    def get_expected_hours(self, employee_id: str, date: datetime) -> tuple:
        schedule = self.get_schedule_by_employee_id(employee_id)
        if not schedule:
            return None, None

        day_name = date.strftime("%A").lower()
        start_key = f"{day_name}_start"
        end_key = f"{day_name}_end"

        start_time = schedule.get(start_key, "OFF")
        end_time = schedule.get(end_key, "OFF")

        return start_time, end_time

    def bulk_import_employees(self, import_csv_path: str) -> bool:
        # Validate input file
        if not os.path.exists(import_csv_path):
            print(f"Error: Import file not found at {import_csv_path}")
            return False

        self._ensure_files_exist()
        imported_count = 0
        errors = []

        # Get existing IDs
        existing_ids = set()
        with open(self.employees_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["employee_id"]:
                    existing_ids.add(row["employee_id"])

        # Import rows
        with open(import_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):
                try:
                    employee_id = row.get("employee_id", "").strip()
                    name = row.get("name", "").strip()
                    person_dir = row.get("person_dir", "").strip()
                    department = row.get("department", "").strip()
                    email = row.get("email", "").strip()

                    # Validate required fields
                    if not employee_id or not name:
                        errors.append(f"Row {row_num}: Missing employee_id or name")
                        continue

                    # Check duplicates
                    if employee_id in existing_ids:
                        errors.append(f"Row {row_num}: Employee ID {employee_id} already exists")
                        continue

                    # Verify face directory exists
                    if person_dir and not os.path.exists(os.path.join(self.db_path, person_dir)):
                        errors.append(f"Row {row_num}: Person directory '{person_dir}' not found in faces_db")
                        continue

                    # Append to CSV
                    with open(self.employees_file, "a", newline="", encoding="utf-8") as out_f:
                        csv.writer(out_f).writerow([employee_id, name, person_dir, department, email])

                    existing_ids.add(employee_id)
                    imported_count += 1

                except Exception as e:
                    errors.append(f"Row {row_num}: {str(e)}")

        # Report results
        if errors:
            print(f"Import completed with {imported_count} employees imported and {len(errors)} errors:")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"Successfully imported {imported_count} employees")

        self.load_employees()
        return imported_count > 0

    def bulk_import_schedules(self, import_csv_path: str) -> bool:
        if not os.path.exists(import_csv_path):
            print(f"Error: Import file not found at {import_csv_path}")
            return False

        self._ensure_files_exist()
        imported_count = 0
        errors = []

        existing_ids = set()
        with open(self.schedules_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["employee_id"]:
                    existing_ids.add(row["employee_id"])

        with open(import_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):
                try:
                    employee_id = row.get("employee_id", "").strip()

                    if not employee_id:
                        errors.append(f"Row {row_num}: Missing employee_id")
                        continue

                    if employee_id in existing_ids:
                        errors.append(f"Row {row_num}: Schedule for employee ID {employee_id} already exists")
                        continue

                    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                    for day in days:
                        start_time = row.get(f"{day}_start", "OFF").strip()
                        end_time = row.get(f"{day}_end", "OFF").strip()

                        if start_time != "OFF" and not self._is_valid_time(start_time):
                            errors.append(f"Row {row_num}: Invalid time format for {day}_start: {start_time}")
                            raise ValueError(f"Invalid time format")

                        if end_time != "OFF" and not self._is_valid_time(end_time):
                            errors.append(f"Row {row_num}: Invalid time format for {day}_end: {end_time}")
                            raise ValueError(f"Invalid time format")

                    with open(self.schedules_file, "a", newline="", encoding="utf-8") as out_f:
                        row_data = [employee_id]
                        for day in days:
                            row_data.append(row.get(f"{day}_start", "OFF").strip())
                            row_data.append(row.get(f"{day}_end", "OFF").strip())
                        row_data.append(row.get("notes", "").strip())
                        csv.writer(out_f).writerow(row_data)

                    existing_ids.add(employee_id)
                    imported_count += 1

                except Exception as e:
                    if not any(e_msg.startswith(f"Row {row_num}") for e_msg in errors):
                        errors.append(f"Row {row_num}: {str(e)}")

        if errors:
            print(f"Import completed with {imported_count} schedules imported and {len(errors)} errors:")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"Successfully imported {imported_count} employee schedules")

        self.load_schedules()
        return imported_count > 0

    def _is_valid_time(self, time_str: str) -> bool:
        try:
            parts = time_str.split(":")
            if len(parts) != 2:
                return False
            hour = int(parts[0])
            minute = int(parts[1])
            return 0 <= hour <= 23 and 0 <= minute <= 59
        except ValueError:
            return False

    def link_face_to_employee(self, employee_id: str, person_dir: str) -> bool:
        if not os.path.exists(os.path.join(self.db_path, person_dir)):
            print(f"Error: Person directory '{person_dir}' not found")
            return False

        self.load_employees()

        updated = False
        temp_rows = []
        with open(self.employees_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["employee_id"] == employee_id:
                    row["person_dir"] = person_dir
                    updated = True
                temp_rows.append(row)

        if updated:
            with open(self.employees_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["employee_id", "name", "person_dir", "department", "email"])
                writer.writeheader()
                writer.writerows(temp_rows)
            self.load_employees()
            return True

        return False

    def get_all_employees(self) -> list:
        if not self.employees:
            self.load_employees()
        return list(self.employees.values())

    def get_employee_by_person_dir(self, person_dir: str) -> dict:
        if not self.employees:
            self.load_employees()
        for emp in self.employees.values():
            if emp.get("person_dir") == person_dir:
                return emp
        return None
