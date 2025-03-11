import pandas as pd
import argparse

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_csv",
                        default="../../../data/phishing4190/phishing4190_2.csv",
                        help='Input csv path to test')
    args = parser.parse_args()
    
    # read screenshot csv
    df_scr = pd.read_csv(args.input_csv)
    # read logo txt
    with open("../object_detection/crop_info.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Strip newline characters
    logos = ["/".join(line.strip().split("/")[-4:]) for line in lines]
    screens = [item.replace("_crop_logo.png", ".png") for item in logos]
    # print(screens)
    filtered_df = df_scr[df_scr['scr_path'].isin(screens)]
    filtered_df["logo_path"] = filtered_df["scr_path"].apply(lambda x: x.replace(".png", "_crop_logo.png"))
    filtered_df.to_csv(args.input_csv.replace(".csv", "_logo.csv"), index=False)