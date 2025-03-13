# %%
import gradio as gr
import numpy as np
import pandas as pd
import pickle

# Load model from file
model_filename = "random_forest_model.pkl"
with open(model_filename, 'rb') as f:
    random_forest_model = pickle.load(f)
    
#import dataset
df = pd.read_csv('apartments_data_enriched_with_new_features.csv')

locations = {
    "Zürich": 261,
    "Kloten": 62,
    "Uster": 198,
    "Illnau-Effretikon": 296,
    "Feuerthalen": 27,
    "Pfäffikon": 177,
    "Ottenbach": 11,
    "Dübendorf": 191,
    "Richterswil": 138,
    "Maur": 195,
    "Embrach": 56,
    "Bülach": 53,
    "Winterthur": 230,
    "Oetwil am See": 157,
    "Russikon": 178,
    "Obfelden": 10,
    "Wald (ZH)": 120,
    "Niederweningen": 91,
    "Dällikon": 84,
    "Buchs (ZH)": 83,
    "Rüti (ZH)": 118,
    "Hittnau": 173,
    "Bassersdorf": 52,
    "Glattfelden": 58,
    "Opfikon": 66,
    "Hinwil": 117,
    "Regensberg": 95,
    "Langnau am Albis": 136,
    "Dietikon": 243,
    "Erlenbach (ZH)": 151,
    "Kappel am Albis": 6,
    "Stäfa": 158,
    "Zell (ZH)": 231,
    "Turbenthal": 228,
    "Oberglatt": 92,
    "Winkel": 72,
    "Volketswil": 199,
    "Kilchberg (ZH)": 135,
    "Wetzikon (ZH)": 121,
    "Zumikon": 160,
    "Weisslingen": 180,
    "Elsau": 219,
    "Hettlingen": 221,
    "Rüschlikon": 139,
    "Stallikon": 13,
    "Dielsdorf": 86,
    "Wallisellen": 69,
    "Dietlikon": 54,
    "Meilen": 156,
    "Wangen-Brüttisellen": 200,
    "Flaach": 28,
    "Regensdorf": 96,
    "Niederhasli": 90,
    "Bauma": 297,
    "Aesch (ZH)": 241,
    "Schlieren": 247,
    "Dürnten": 113,
    "Unterengstringen": 249,
    "Gossau (ZH)": 115,
    "Oberengstringen": 245,
    "Schleinikon": 98,
    "Aeugst am Albis": 1,
    "Rheinau": 38,
    "Höri": 60,
    "Rickenbach (ZH)": 225,
    "Rafz": 67,
    "Adliswil": 131,
    "Zollikon": 161,
    "Urdorf": 250,
    "Hombrechtikon": 153,
    "Birmensdorf (ZH)": 242,
    "Fehraltorf": 172,
    "Weiach": 102,
    "Männedorf": 155,
    "Küsnacht (ZH)": 154,
    "Hausen am Albis": 4,
    "Hochfelden": 59,
    "Fällanden": 193,
    "Greifensee": 194,
    "Mönchaltorf": 196,
    "Dägerlen": 214,
    "Thalheim an der Thur": 39,
    "Uetikon am See": 159,
    "Seuzach": 227,
    "Uitikon": 248,
    "Affoltern am Albis": 2,
    "Geroldswil": 244,
    "Niederglatt": 89,
    "Thalwil": 141,
    "Rorbas": 68,
    "Pfungen": 224,
    "Weiningen (ZH)": 251,
    "Bubikon": 112,
    "Neftenbach": 223,
    "Mettmenstetten": 9,
    "Otelfingen": 94,
    "Flurlingen": 29,
    "Stadel": 100,
    "Grüningen": 116,
    "Henggart": 31,
    "Dachsen": 25,
    "Bonstetten": 3,
    "Bachenbülach": 51,
    "Horgen": 295
}

# Define the core prediction function
def predict_apartment(rooms, area, town, tax_income, luxurious, temporary, furnished, room_per_m2, zurich_city):
    bfs_number = locations[town]
    df1 = df[df['bfs_number']==bfs_number].copy()
    df1.reset_index(inplace=True)
    df1.loc[0, 'rooms'] = rooms
    df1.loc[0, 'area'] = area
    df1.loc[0, 'tax_income'] = tax_income
    df1.loc[0, 'luxurious'] = luxurious
    df1.loc[0, 'temporary'] = temporary
    df1.loc[0, 'furnished'] = furnished
    df1.loc[0, 'room_per_m2'] = room_per_m2
    df1.loc[0, 'zurich_city'] = zurich_city
    
    if len(df1) > 1:  # if there are more than one record with the same bfs_number, calculate the mean price
        df1[0, 'price'] = df1['price'].mean()
        
    prediction = random_forest_model.predict(df1[['rooms', 'area', 'pop', 'pop_dens', 'frg_pct', 'emp', 'tax_income', 'room_per_m2', 'luxurious', 'temporary', 'furnished', 'area_cat_ecoded', 'zurich_city']])
    return np.round(prediction[0], 0)

# Create the Gradio interface
demo = gr.Interface(
    fn=predict_apartment,
    inputs=[
        gr.Number(label="Rooms"),
        gr.Number(label="Area"),
        gr.Dropdown(choices=list(locations.keys()), label="Town"),
        gr.Number(label="Tax Income"),
        gr.Checkbox(label="Luxurious"),
        gr.Checkbox(label="Temporary"),
        gr.Checkbox(label="Furnished")
    ],
    outputs=gr.Number(),
    examples=[
        [4.5, 120, "Dietlikon", 90000, 2000, True, False, True],
        [3.5, 60, "Winterthur", 85000, 1500, False, True, False],
        [2.5, 40, "Zürich", 110000, 5000, True, True, True],
    ]
)

demo.launch()