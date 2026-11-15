import sqlite3

connection = sqlite3.connect(
    "file:data/seven.db?mode=ro",
    uri=True,
)

tables = connection.execute(
    "SELECT name FROM sqlite_master "
    "WHERE type='table' ORDER BY name"
).fetchall()

print("TABLES:")
for table in tables:
    print("-", table[0])

print("\nALEMBIC VERSION:")

if any(table[0] == "alembic_version" for table in tables):
    print(connection.execute(
        "SELECT * FROM alembic_version"
    ).fetchall())
else:
    print("Not initialized")

connection.close()
