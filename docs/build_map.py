"""
Пересборка интерактивной карты привлекательности локаций из данных репозитория.

Повторяет пайплайн из notebooks/main_file.ipynb, но берёт всё из data/ —
без обращений к внешним API, поэтому запускается где угодно и всегда даёт
одинаковый результат.

Запуск:  python build_map.py
Результат: docs/index.html
"""

import glob
import re
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt

ROOT = Path(__file__).parent
OUT = ROOT / "docs" / "index.html"


# --------------------------------------------------------------------------
# 1. Сетка
# --------------------------------------------------------------------------

grid = pd.read_csv(ROOT / "data/grid/grid_with_walkability_and_parkings.csv")
grid["geometry"] = grid["geometry"].apply(wkt.loads)
grid = gpd.GeoDataFrame(grid, geometry="geometry", crs="EPSG:4326")
print(f"Ячеек сетки: {len(grid)}")


# --------------------------------------------------------------------------
# 2. Метро: ближайшая станция и её пассажиропоток
# --------------------------------------------------------------------------

metro = pd.read_csv(ROOT / "data/metro/metro_stations_with_passenger_flow.csv")
metro = metro.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

# Среднесуточный поток: годовой вход + выход, делённые на 365.
metro["metro_flow"] = (metro["IncomingPassengers"] + metro["OutgoingPassengers"]) / 365

# Расстояние до ближайшей станции. Пересчитываем градусы в метры через
# локальное приближение: по широте 111 320 м, по долготе — с поправкой на cos(lat).
LAT_M = 111_320
lat0 = np.deg2rad(grid["center_lat"].mean())
LON_M = LAT_M * np.cos(lat0)

cell_xy = np.c_[grid["center_lon"] * LON_M, grid["center_lat"] * LAT_M]
metro_xy = np.c_[metro["longitude"] * LON_M, metro["latitude"] * LAT_M]

dist = np.sqrt(((cell_xy[:, None, :] - metro_xy[None, :, :]) ** 2).sum(axis=2))
nearest = dist.argmin(axis=1)

grid["distance_to_metro_new"] = dist.min(axis=1)
grid["nearest_metro"] = metro["station"].values[nearest]
grid["metro_flow"] = metro["metro_flow"].values[nearest]

print(f"Медиана расстояния до метро: {grid['distance_to_metro_new'].median():,.0f} м"
      .replace(",", " "))


# --------------------------------------------------------------------------
# 3. Аренда: парсинг цены из заголовков ЦИАН, станция — из адреса
# --------------------------------------------------------------------------

cian = pd.concat(
    [pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "data/cian/*.csv")))],
    ignore_index=True,
).drop_duplicates(subset=["link"])

# Цена лежит внутри заголовка: «Своб. назнач. 118 м² за 224 200 руб./мес.»
price_pattern = re.compile(r"за\s*([\d\s]+)\s*руб")


def extract_price(text: str):
    if not isinstance(text, str):
        return None
    m = price_pattern.search(text.replace("\xa0", " "))
    return int(m.group(1).replace(" ", "")) if m else None


# Станция — первая строка адреса до разделителя «⋅».
def extract_station(text: str):
    if not isinstance(text, str):
        return None
    return text.split("\n")[0].strip().lower()


cian["price"] = cian["title"].apply(extract_price)
cian["station"] = cian["address"].apply(extract_station)
cian = cian.dropna(subset=["price", "station"])

# Отсечение выбросов по правилу полутора межквартильных размахов.
q1, q3 = cian["price"].quantile([0.25, 0.75])
iqr = q3 - q1
before = len(cian)
cian = cian[(cian["price"] >= q1 - 1.5 * iqr) & (cian["price"] <= q3 + 1.5 * iqr)]
print(f"Объявлений ЦИАН: {before:,} → после отсечения выбросов {len(cian):,}"
      .replace(",", " "))

rent = cian.groupby("station")["price"].mean().rename("avg_price_rent")
grid = grid.merge(rent, left_on="nearest_metro", right_index=True, how="left")

# Если до метро больше 5 км, привязка к его ставке аренды бессмысленна.
grid.loc[grid["distance_to_metro_new"] >= 5000, "avg_price_rent"] = np.nan
print(f"Ячеек с оценкой аренды: {grid['avg_price_rent'].notna().sum()} из {len(grid)}")


# --------------------------------------------------------------------------
# 4. Пешеходный поток
# --------------------------------------------------------------------------


def norm(s: pd.Series) -> pd.Series:
    return ((s - s.min()) / (s.max() - s.min())).fillna(0)


grid["metro_flow_norm"] = norm(grid["metro_flow"])
grid["parking_count_norm"] = norm(grid["parking_count"])
grid["poi_count_norm"] = norm(grid["poi_count"])
grid["distance_to_metro_log"] = np.log1p(grid["distance_to_metro_new"])


def foot_traffic(a: float, b: float, g: float) -> pd.Series:
    return (
        a * (grid["metro_flow_norm"] / (grid["distance_to_metro_log"] + 0.01))
        + b * grid["parking_count_norm"]
        + g * np.sqrt(grid["poi_count_norm"])
    ).fillna(0)


# Веса подбираются так, чтобы расчётный поток максимально коррелировал
# с числом отзывов о заведениях — единственным доступным прокси реальной
# посещаемости.
best_r, best = 0.0, None
for a in np.linspace(0.4, 0.8, 9):
    for b in np.linspace(0.1, 0.4, 7):
        for g in np.linspace(0.1, 0.4, 7):
            r = foot_traffic(a, b, g).corr(grid["total_reviews_count"])
            if r > best_r:
                best_r, best = r, (a, b, g)

grid["daily_foot_traffic"] = foot_traffic(*best)
print(f"Веса потока: α={best[0]:.2f}, β={best[1]:.2f}, γ={best[2]:.2f} "
      f"· корреляция с числом отзывов r={best_r:.3f}")

# Перевод безразмерного индекса в «людей в день» через максимальный поток метро.
scale = grid["metro_flow"].max() * 1.5
grid["daily_foot_traffic_real"] = (
    grid["daily_foot_traffic"] / grid["daily_foot_traffic"].max() * scale
)


# --------------------------------------------------------------------------
# 5. Индекс привлекательности
# --------------------------------------------------------------------------

for col in [
    "daily_foot_traffic_real",
    "average_cafe_rating",
    "poi_count",
    "chain_cafe_count",
    "cafe_count",
    "avg_price_rent",
]:
    grid[col + "_norm"] = norm(grid[col])

grid["attractiveness_index"] = (
    0.40 * grid["daily_foot_traffic_real_norm"]   # пешеходный поток
    + 0.25 * grid["average_cafe_rating_norm"]     # средний рейтинг кафе в зоне
    + 0.15 * grid["poi_count_norm"]               # насыщенность инфраструктурой
    - 0.30 * grid["chain_cafe_count_norm"]        # конкуренция со стороны сетей
    - 0.30 * grid["cafe_count_norm"]              # общая плотность конкурентов
    - 0.20 * grid["avg_price_rent_norm"]          # стоимость аренды
)

p1, p99 = np.percentile(grid["attractiveness_index"], [1, 99])
idx = grid["attractiveness_index"].clip(p1, p99)
grid["index_final"] = ((idx - idx.min()) / (idx.max() - idx.min())).round(3)

print("\nРаспределение индекса:")
print(grid["index_final"].describe().round(3).to_string())


# --------------------------------------------------------------------------
# 6. Расстояние до центра — проверка вывода про 5–10 км
# --------------------------------------------------------------------------

CENTER = (55.7558, 37.6176)  # Красная площадь
grid["dist_center_km"] = np.sqrt(
    ((grid["center_lon"] - CENTER[1]) * LON_M) ** 2
    + ((grid["center_lat"] - CENTER[0]) * LAT_M) ** 2
) / 1000

bands = pd.cut(grid["dist_center_km"], [0, 2, 5, 10, 15, 100],
               labels=["0–2 км", "2–5 км", "5–10 км", "10–15 км", "15+ км"])
by_band = grid.groupby(bands, observed=True).agg(
    ячеек=("index_final", "size"),
    индекс=("index_final", "mean"),
    аренда=("avg_price_rent", "median"),
    кафе=("cafe_count", "median"),
).round(2)
print("\nСредний индекс по удалённости от центра:")
print(by_band.to_string())

top = grid.nlargest(50, "index_final")
print(f"\nТоп-50 ячеек: медиана удалённости от центра {top['dist_center_km'].median():.1f} км, "
      f"диапазон {top['dist_center_km'].min():.1f}–{top['dist_center_km'].max():.1f} км")


# --------------------------------------------------------------------------
# 7. Карта
# --------------------------------------------------------------------------

gdf = grid[[
    "grid_id", "geometry", "index_final", "nearest_metro",
    "cafe_count", "chain_cafe_count", "avg_price_rent", "dist_center_km",
]].copy()
gdf["avg_price_rent"] = gdf["avg_price_rent"].round(0)
gdf["dist_center_km"] = gdf["dist_center_km"].round(1)

# Подложка OpenStreetMap: CartoDB с 2025 года требует API-ключ, а карта
# должна открываться у кого угодно без регистрации.
m = folium.Map(location=[55.75, 37.62], zoom_start=10, tiles="OpenStreetMap")

folium.Choropleth(
    geo_data=gdf,
    data=gdf,
    columns=["grid_id", "index_final"],
    key_on="feature.properties.grid_id",
    fill_color="YlOrRd",
    fill_opacity=0.75,
    line_opacity=0.15,
    nan_fill_opacity=0,
    legend_name="Индекс привлекательности локации (0–1)",
).add_to(m)

folium.GeoJson(
    gdf,
    style_function=lambda _: {"fillOpacity": 0, "color": "transparent", "weight": 0},
    highlight_function=lambda _: {"fillOpacity": 0.25, "color": "#0b0b0b", "weight": 1},
    tooltip=folium.GeoJsonTooltip(
        fields=["index_final", "nearest_metro", "cafe_count",
                "chain_cafe_count", "avg_price_rent", "dist_center_km"],
        aliases=["Индекс:", "Ближайшее метро:", "Кафе в квадрате:",
                 "Из них сетевых:", "Аренда, руб./мес.:", "До центра, км:"],
        localize=True,
        sticky=False,
    ),
).add_to(m)

title = """
<div style="position: fixed; top: 12px; left: 60px; z-index: 9999;
            background: #fcfcfb; padding: 12px 16px; border-radius: 8px;
            border: 1px solid rgba(11,11,11,.12); max-width: 380px;
            font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
  <div style="font-size: 15px; font-weight: 600; color: #0b0b0b;">
    Где в Москве открывать кафе
  </div>
  <div style="font-size: 12px; color: #52514e; margin-top: 6px; line-height: 1.45;">
    1709 квадратов 800×800 м внутри МКАД. Индекс учитывает пешеходный поток,
    конкуренцию, аренду и инфраструктуру. Наведите на квадрат — покажет детали.
  </div>
  <div style="font-size: 11px; color: #898781; margin-top: 8px;">
    Учебный проект НИУ ВШЭ, 2025 · данные: портал открытых данных Москвы,
    2ГИС, ЦИАН, OpenStreetMap
  </div>
</div>
"""
m.get_root().html.add_child(folium.Element(title))

OUT.parent.mkdir(exist_ok=True)
m.save(str(OUT))
print(f"\nКарта сохранена: {OUT}  ({OUT.stat().st_size / 1024:.0f} КБ)")
