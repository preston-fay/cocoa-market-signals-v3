#!/usr/bin/env python3
"""Test alignment between events and prices"""
import json

# Load the data
with open('data/processed/detailed_events.json', 'r') as f:
    events = json.load(f)

with open('data/processed/timeline_data_complete.json', 'r') as f:
    timeline = json.load(f)

# Check date formats
print("First 5 event dates:")
for e in events[:5]:
    print(f"  {e['date']} -> {e['price_change']*100:.1f}%")

print("\nFirst 5 price dates:")
for p in timeline['prices'][:5]:
    print(f"  {p['date']} -> {p['return']:.1f}%")

# Find exact matches
price_date_map = {p['date']: p['return'] for p in timeline['prices']}
matches = 0
for e in events:
    if e['date'] in price_date_map:
        matches += 1
        print(f"\nMatch found: {e['date']}")
        print(f"  Event return: {e['price_change']*100:.1f}%")
        print(f"  Price return: {price_date_map[e['date']]:.1f}%")
        if matches > 3:
            break

print(f"\nTotal matches: {sum(1 for e in events if e['date'] in price_date_map)}/{len(events)}")