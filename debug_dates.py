#!/usr/bin/env python3
"""Debug date alignment issues"""
import json

# Load the data
events = json.load(open('data/processed/detailed_events.json'))
dashboard = json.load(open('data/processed/real_dashboard_data.json'))

# Check date formats
print("Event dates sample:")
for e in events[:5]:
    print(f"  {e['date']}")

print("\nDashboard prediction dates sample:")
for p in dashboard['predictions'][:5]:
    print(f"  {p['date']}")

# Check if any events have matching dates
pred_dates = {p['date'].split('T')[0] for p in dashboard['predictions']}
event_dates = {e['date'] for e in events}

print(f"\nUnique event dates: {len(event_dates)}")
print(f"Unique prediction dates: {len(pred_dates)}")

# Find events that should match
matching = []
for e in events:
    if e['date'] in pred_dates:
        matching.append(e['date'])

print(f"\nEvents with matching prediction dates: {len(matching)}")
if matching:
    print("Examples:", matching[:5])