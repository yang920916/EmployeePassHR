# 員工通｜人資後台 Web（每日出勤 + 整月薪資）

## 入口
- /admin-web/login
- /admin-web/dashboard（每日）
- /admin-web/payroll（整月）

## API
- GET /admin/attendance/daily?date=YYYY-MM-DD
- GET /admin/attendance/export/csv?date=YYYY-MM-DD
- GET /admin/payroll/monthly?month=YYYY-MM
- GET /admin/payroll/export/csv?month=YYYY-MM
- PATCH /admin/users/{id}/hourly_rate

## 加班倍率
環境變數：OVERTIME_MULTIPLIER（預設 1.33）
