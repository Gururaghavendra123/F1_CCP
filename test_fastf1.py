import fastf1

fastf1.Cache.enable_cache('f1_cache')

session = fastf1.get_session(2023, 1, 'Q')
session.load()

print("Event:", session.event)

results = session.results
print(results[['Abbreviation', 'TeamName', 'Position', 'Q1', 'Q2', 'Q3']].head())
print (results.head())
results.to_csv("bahrain_2023_qualifying.csv", index=False)
print("✅ Saved results to bahrain_2023_qualifying.csv")