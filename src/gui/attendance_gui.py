import os
import sys
import csv
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import FreeSimpleGUI as sg
from src.attendance.employee_manager import EmployeeManager
from src.attendance.attendance_analyzer import AttendanceAnalyzer
from src.attendance.report_generator import ReportGenerator

BG_COLOR = "#1A1414"
PAPER_COLOR = "#272121"
TEXT_COLOR = "#ffffff"
TEXT_SEC_COLOR = "#b3b3b3"
ACCENT_COLOR = "#90caf9"

sg.LOOK_AND_FEEL_TABLE['MaterialDark'] = {
    'BACKGROUND': BG_COLOR,
    'TEXT': TEXT_COLOR,
    'INPUT': PAPER_COLOR,
    'TEXT_INPUT': TEXT_COLOR,
    'SCROLL': PAPER_COLOR,
    'BUTTON': (TEXT_COLOR, PAPER_COLOR),
    'PROGRESS': (ACCENT_COLOR, PAPER_COLOR),
    'BORDER': 1,
    'SLIDER_DEPTH': 0,
    'PROGRESS_DEPTH': 0
}
sg.theme('MaterialDark')


class AttendanceGUI:
    def __init__(self):
        self.attendance_dir = "src/attendance"
        self.employee_manager = EmployeeManager(self.attendance_dir)
        self.analyzer = AttendanceAnalyzer(self.attendance_dir)
        self.report_gen = ReportGenerator(self.attendance_dir)
        self.window = None

    def _frame(self, title, layout):
        return sg.Frame(title, layout, font='Helvetica 11 bold', title_color=ACCENT_COLOR,
                       background_color=PAPER_COLOR, pad=(10, 10), border_width=0, expand_x=True)

    def _text(self, label=""):
        return sg.Text(label, text_color=TEXT_SEC_COLOR)

    def _input(self, default="", key="", size=(15, 1)):
        return sg.InputText(default, key=key, size=size, background_color=BG_COLOR, text_color=TEXT_COLOR)

    def _listbox(self, key, rows=15):
        return sg.Listbox([], size=(50, rows), key=key, text_color=TEXT_COLOR,
                         background_color=BG_COLOR, highlight_background_color=ACCENT_COLOR)

    def build_layout(self):
        today = datetime.now()
        week_ago = today - timedelta(days=7)

        employees_tab = [
            [sg.Text("Employee List", font='Helvetica 12 bold', text_color=ACCENT_COLOR)],
            [self._listbox('EMPLOYEE_LIST', 15)],
            [sg.Button("Refresh", key="REFRESH_EMPLOYEES"), sg.Button("Import CSV", key="IMPORT_EMPLOYEES")],
        ]

        schedules_tab = [
            [sg.Text("Schedule Management", font='Helvetica 12 bold', text_color=ACCENT_COLOR)],
            [self._text("Employee ID:"), self._input(key='SCHED_EMP_ID')],
            [self._text("Monday (Start):"), self._input('09:00', 'MON_START'),
             self._text("End:"), self._input('17:00', 'MON_END')],
            [sg.Button("Import Schedules", key="IMPORT_SCHEDULES"),
             sg.Button("Set Full Week", key="SET_WEEK")],
        ]

        attendance_tab = [
            [sg.Text("Attendance Analysis", font='Helvetica 12 bold', text_color=ACCENT_COLOR)],
            [self._text("Date:"), self._input(today.strftime("%Y-%m-%d"), 'ANALYZE_DATE'),
             sg.Button("Analyze", key="ANALYZE_DATE_BTN")],
            [self._listbox('ATTENDANCE_LIST', 15)],
            [sg.Text("", key='ATTENDANCE_STATS', text_color=ACCENT_COLOR)],
        ]

        reports_tab = [
            [sg.Text("Report Generation", font='Helvetica 12 bold', text_color=ACCENT_COLOR)],
            [sg.Text("Date Range:", text_color=TEXT_SEC_COLOR)],
            [self._text("From:"), self._input(week_ago.strftime("%Y-%m-%d"), 'REPORT_START'),
             self._text("To:"), self._input(today.strftime("%Y-%m-%d"), 'REPORT_END')],
            [sg.Button("Generate Daily", key="GEN_DAILY"),
             sg.Button("Generate Weekly", key="GEN_WEEKLY"),
             sg.Button("Export CSV", key="EXPORT_CSV")],
            [self._listbox('REPORT_LIST', 12)],
            [sg.Text("Status: Ready", key='REPORT_STATUS', text_color=ACCENT_COLOR)],
        ]

        flagged_tab = [
            [sg.Text("Flagged Records", font='Helvetica 12 bold', text_color=ACCENT_COLOR)],
            [self._text("View exceptions (Late, Absent, Unknown)")],
            [self._listbox('FLAGGED_LIST', 15)],
            [sg.Button("Refresh Flagged", key="REFRESH_FLAGGED"),
             sg.Button("Export Flagged", key="EXPORT_FLAGGED")],
        ]

        tabs = [
            [sg.TabGroup([
                [sg.Tab('Employees', employees_tab, background_color=BG_COLOR)],
                [sg.Tab('Schedules', schedules_tab, background_color=BG_COLOR)],
                [sg.Tab('Attendance', attendance_tab, background_color=BG_COLOR)],
                [sg.Tab('Reports', reports_tab, background_color=BG_COLOR)],
                [sg.Tab('Flagged', flagged_tab, background_color=BG_COLOR)],
            ], background_color=BG_COLOR, tab_background_color=PAPER_COLOR)]
        ]

        return [
            [sg.Text("Attendance Management System", font='Helvetica 16 bold', text_color=TEXT_COLOR)],
            tabs,
            [sg.Button("Close", key="EXIT", button_color=(TEXT_COLOR, "#f44336"))]
        ]

    def _format_record(self, record, include_deviation=False):
        status = record['status']
        icon = "✓" if status == "ON_TIME" else "✗" if status in ["LATE", "ABSENT"] else "?"
        line = f"[{icon}] {record['name']:20} {status:10} (Exp: {record['expected_start']}, Act: {record['actual_start']})"
        if include_deviation and record.get('deviation_minutes') != "N/A":
            line += f" +{record['deviation_minutes']}m"
        return line

    def _show_error(self, widget_key, message):
        self.window[widget_key].update([f"Error: {message}"])

    def refresh_employees(self):
        employees = self.employee_manager.get_all_employees()
        emp_list = [f"{e['employee_id']}: {e['name']} ({e['department']})" for e in employees]
        self.window['EMPLOYEE_LIST'].update(emp_list)

    def analyze_attendance(self, date_str):
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            start_date = datetime.combine(date_obj.date(), datetime.min.time())
            end_date = start_date + timedelta(days=1)

            analysis = self.analyzer.analyze_date_range(start_date, end_date)
            if not analysis or date_str not in analysis:
                self.window['ATTENDANCE_LIST'].update(["No attendance data for this date"])
                return

            daily = analysis[date_str]
            stats = self.analyzer.get_attendance_summary_stats(daily)

            records = [self._format_record(r, True) for r in daily.values()]
            self.window['ATTENDANCE_LIST'].update(records)

            stats_text = f"On-time: {stats['ON_TIME']} | Late: {stats['LATE']} | Absent: {stats['ABSENT']} | Off: {stats['OFF']} | Unknown: {stats['UNKNOWN']}"
            self.window['ATTENDANCE_STATS'].update(stats_text)
        except Exception as e:
            self._show_error('ATTENDANCE_LIST', str(e))

    def generate_reports(self, start_str, end_str, report_type):
        try:
            start = datetime.strptime(start_str, "%Y-%m-%d")
            end = datetime.strptime(end_str, "%Y-%m-%d") + timedelta(days=1)

            if report_type in ["daily", "all"]:
                for i in range((end - start).days):
                    self.report_gen.generate_daily_summary(start + timedelta(days=i))

            if report_type in ["weekly", "all"]:
                self.report_gen.generate_weekly_summary(start, end)

            self.window['REPORT_STATUS'].update("Reports generated successfully", text_color="#4CAF50")
            self._refresh_reports(start, end)
        except Exception as e:
            self.window['REPORT_STATUS'].update(f"Error: {str(e)}", text_color="#f44336")

    def _refresh_reports(self, start, end):
        try:
            flagged = self.report_gen.generate_flagged_report(start, end)
            records = [f"{f['date']} | {self._format_record(f)}" for f in flagged]
            self.window['REPORT_LIST'].update(records or ["No flagged records"])
        except Exception as e:
            self._show_error('REPORT_LIST', str(e))

    def _refresh_flagged(self):
        try:
            today = datetime.now()
            flagged = self.report_gen.generate_flagged_report(today - timedelta(days=7), today)
            records = [f"{f['date']} | {f['name']:20} {f['status']:10}" for f in flagged]
            self.window['FLAGGED_LIST'].update(records or ["No exceptions found"])
        except Exception as e:
            self._show_error('FLAGGED_LIST', str(e))

    def _export_flagged(self):
        filepath = sg.popup_get_file("Save flagged report as:", save_as=True,
                                     file_types=(("CSV files", "*.csv"), ("ALL Files", "*.*")))
        if not filepath:
            return

        try:
            today = datetime.now()
            flagged = self.report_gen.generate_flagged_report(today - timedelta(days=30), today)
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['date', 'employee_id', 'name', 'expected_start', 'actual_start', 'status', 'deviation_minutes', 'notes'])
                writer.writeheader()
                writer.writerows(flagged)
            sg.popup_ok(f"Flagged report exported to {filepath}")
        except Exception as e:
            sg.popup_error(f"Export failed: {str(e)}")

    def _import_file_dialog(self, title):
        return sg.popup_get_file(title, file_types=(("CSV files", "*.csv"), ("ALL Files", "*.*")))

    def run(self):
        self.window = sg.Window('Attendance Management System', self.build_layout(),
                               background_color=BG_COLOR, size=(900, 700), finalize=True)

        self.refresh_employees()
        self._refresh_flagged()

        while True:
            event, values = self.window.read()

            if event == sg.WIN_CLOSED or event == 'EXIT':
                break

            if event == 'REFRESH_EMPLOYEES':
                self.refresh_employees()
            elif event == 'IMPORT_EMPLOYEES':
                filepath = self._import_file_dialog("Select employee CSV to import:")
                if filepath:
                    self.employee_manager.bulk_import_employees(filepath)
                    self.refresh_employees()
            elif event == 'IMPORT_SCHEDULES':
                filepath = self._import_file_dialog("Select schedules CSV to import:")
                if filepath:
                    self.employee_manager.bulk_import_schedules(filepath)
                    sg.popup_ok("Schedules imported successfully")
            elif event == 'ANALYZE_DATE_BTN':
                self.analyze_attendance(values['ANALYZE_DATE'])
            elif event in ['GEN_DAILY', 'GEN_WEEKLY']:
                report_type = 'daily' if event == 'GEN_DAILY' else 'weekly'
                self.generate_reports(values['REPORT_START'], values['REPORT_END'], report_type)
            elif event == 'EXPORT_CSV':
                filepath = sg.popup_get_file("Save report as:", save_as=True,
                                            file_types=(("CSV files", "*.csv"), ("ALL Files", "*.*")))
                if filepath:
                    try:
                        today = datetime.now()
                        self.report_gen.export_report(today - timedelta(days=7), today, filepath)
                        sg.popup_ok(f"Report exported to {filepath}")
                    except Exception as e:
                        sg.popup_error(f"Export failed: {str(e)}")
            elif event == 'REFRESH_FLAGGED':
                self._refresh_flagged()
            elif event == 'EXPORT_FLAGGED':
                self._export_flagged()

        self.window.close()


if __name__ == "__main__":
    gui = AttendanceGUI()
    gui.run()

