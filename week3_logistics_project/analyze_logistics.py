from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT, VIZ = ROOT / "outputs", ROOT / "visualizations"
OUT.mkdir(exist_ok=True); VIZ.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "logistics_hypothetical_data.csv")
numeric = ["Distance_km","Shipment_Volume_kg","Transit_Time_days","Transport_Cost_INR"]

pd.DataFrame({"Metric":["Rows","Columns","Missing cells","Duplicate IDs"],
              "Value":[len(df),len(df.columns),int(df.isna().sum().sum()),int(df.Shipment_ID.duplicated().sum())]}).to_csv(OUT/"dataset_overview.csv",index=False)
df[numeric].describe().T.to_csv(OUT/"descriptive_statistics.csv")
df[numeric].agg(["mean","median","std","min","max"]).T.to_csv(OUT/"central_tendency_and_spread.csv")
pd.DataFrame([["Total shipments",len(df)],["On-time rate (%)",round(df.On_Time_Delivery.eq("Yes").mean()*100,2)],
 ["Average transit time (days)",round(df.Transit_Time_days.mean(),2)],["Median transit time (days)",round(df.Transit_Time_days.median(),2)],
 ["Average transport cost (INR)",round(df.Transport_Cost_INR.mean(),2)]],columns=["KPI","Value"]).to_csv(OUT/"kpi_summary.csv",index=False)
for col in ["Transport_Mode","Destination","Month"]:
    summary=df.groupby(col).agg(Shipments=("Shipment_ID","count"),Avg_Transit_Days=("Transit_Time_days","mean"),
       Avg_Cost_INR=("Transport_Cost_INR","mean"),On_Time_Rate_Pct=("On_Time_Delivery",lambda s:s.eq("Yes").mean()*100)).round(2)
    summary.to_csv(OUT/f"performance_by_{col.lower()}.csv")
corr=df[numeric].corr(); corr.to_csv(OUT/"correlation_matrix.csv")
plt.figure(figsize=(8,5)); plt.hist(df.Transit_Time_days,bins=14,edgecolor="black"); plt.title("Distribution of Shipment Transit Time"); plt.xlabel("Transit time (days)"); plt.ylabel("Shipments"); plt.tight_layout(); plt.savefig(VIZ/"transit_time_distribution.png",dpi=160); plt.close()
plt.figure(figsize=(8,5)); df.groupby("Transport_Mode").Transport_Cost_INR.mean().reindex(["Road","Rail","Air"]).plot(kind="bar"); plt.title("Average Transport Cost by Mode"); plt.xlabel("Mode"); plt.ylabel("Average cost (INR)"); plt.tight_layout(); plt.savefig(VIZ/"average_cost_by_mode.png",dpi=160); plt.close()
plt.figure(figsize=(8,5)); df.groupby("Month").Transit_Time_days.mean().reindex(["June","July","August","September"]).plot(marker="o"); plt.title("Average Transit Time by Month"); plt.xlabel("Month"); plt.ylabel("Average transit time (days)"); plt.tight_layout(); plt.savefig(VIZ/"monthly_transit_trend.png",dpi=160); plt.close()
plt.figure(figsize=(8,5)); plt.scatter(df.Distance_km,df.Transport_Cost_INR,alpha=.75); plt.title("Distance vs Transport Cost"); plt.xlabel("Distance (km)"); plt.ylabel("Cost (INR)"); plt.tight_layout(); plt.savefig(VIZ/"distance_vs_cost.png",dpi=160); plt.close()
plt.figure(figsize=(8,5)); plt.scatter(df.Shipment_Volume_kg,df.Transport_Cost_INR,alpha=.75); plt.title("Shipment Volume vs Transport Cost"); plt.xlabel("Volume (kg)"); plt.ylabel("Cost (INR)"); plt.tight_layout(); plt.savefig(VIZ/"volume_vs_cost.png",dpi=160); plt.close()
plt.figure(figsize=(7,5)); plt.imshow(corr.values,vmin=-1,vmax=1,aspect="auto"); plt.xticks(range(len(numeric)),numeric,rotation=35,ha="right"); plt.yticks(range(len(numeric)),numeric); plt.colorbar(label="Correlation"); plt.title("Correlation Matrix"); plt.tight_layout(); plt.savefig(VIZ/"correlation_matrix.png",dpi=160); plt.close()
print("Analysis completed."); print("Shipments:",len(df)); print("On-time rate:",round(df.On_Time_Delivery.eq("Yes").mean()*100,2),"%")
