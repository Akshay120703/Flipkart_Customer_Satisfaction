import pandas as pd
df = pd.read_excel("C:\Users\aksha\Downloads\1800+ Talent Acquisition Database by SoarX.xlsx")
df[['Linkedin URL']].rename(columns={'Linkedin URL': 'linkedinUrl'}).to_csv("linkedin_profiles.csv", index=False)
