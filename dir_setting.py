import numpy as np
import pandas as pd
import os


def final_df(Ylabel, ylabel, dataset, arch, hue_order, mode='figure', opt='average',
             exp_type_order=None, f_name=None, root=None, read_dir=None):
    df_list = []

    # change the order of the graphs by changing this exp_type order
    exp_type_order2 = []
    exp_dict = {}
    for type in exp_type_order:
        if type == "0.2_sym":
            exp_type_order2.append("sym_20")
        elif type == "0.5_sym":
            exp_type_order2.append("sym_50")
        elif type == "0.8_sym":
            exp_type_order2.append("sym_80")
        elif type == "0.9_sym":
            exp_type_order2.append("sym_90")
        elif type == "0.4_asym":
            exp_type_order2.append("asym_40")
    exp_dict = {"0.2_sym": "Symmetric 20%", "0.5_sym": "Symmetric 50%",
                "0.8_sym": "Symmetric 80%", "0.9_sym": "Symmetric 90%",
                "0.4_asym": "Asymmetric 40%"}

    if mode == 'figure':
        def make_df_plus(method, base_addr, dataset, exp_type):
            if ylabel == 'Accuracy':
                df1 = pd.read_csv(
                    f"{base_addr}/{dataset}_{exp_type}_acc.txt", delim_whitespace=True
                )
            elif ylabel == 'Number_of_labeled_samples':
                df1 = pd.read_csv(
                    f"{base_addr}/{dataset}_{exp_type}_stats1.txt", delim_whitespace=True
                )
                df1[ylabel] = [i/500 for i in df1[ylabel]]
            elif ylabel == "AUC":
                df1 = pd.read_csv(
                    f"{base_addr}/{dataset}_{exp_type}_stats1.txt", delim_whitespace=True
                )
            elif ylabel == "plabel_acc":
                df1 = pd.read_csv(
                    f"{base_addr}/{dataset}_{exp_type}_stats1.txt", delim_whitespace=True
                )
                df1[ylabel] = [i*100 for i in df1[ylabel]]

            df2 = pd.DataFrame(
                {"Epoch": np.array(df1["Epoch"], dtype='int32'),
                 'Noise Type': exp_dict[exp_type], Ylabel: df1[ylabel],
                 'Method': method}
            )
            df_list.extend([df2])
        process = make_df_plus
    elif mode == 'last_ten_epoch':
        df = pd.DataFrame(
            np.arange(
                len(exp_dict) * len(hue_order),
                dtype=np.float).reshape(len(hue_order), (len(exp_dict))),
            index=list(hue_order), columns=list(map(lambda key: exp_dict[key], exp_type_order))
        )

        def last_ten_epoch(method, base_addr, dataset, exp_type):
            total = 0
            df1 = pd.read_csv(
                f"{base_addr}/{dataset}_{exp_type}_acc.txt", delim_whitespace=True
            )
            if opt == 'average':
                total += df1[ylabel][-10:]
                from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
                # general rounding
                avg = Decimal(np.mean(total)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                df.at[method, exp_dict[exp_type]] = avg
            elif opt == 'best':
                df.at[method, exp_dict[exp_type]] = np.max(df1[ylabel])
        process = last_ten_epoch
    else:
        raise NotImplementedError

    for exp_type, exp_type2 in exp_type_order, exp_type_order2:
        method = 'DivideMix'
        if method in hue_order:
            base_addr = "{}{}/{}_{}_{}_{}".format(
                root, read_dir[method], dataset, method, arch, exp_type2)
            process(method, base_addr, dataset, exp_type)

        method = 'Proposed'
        if method in hue_order:
            base_addr = "{}{}/{}_UPL_{}_{}".format(
                root, read_dir[method], dataset, arch, exp_type2)
            process(method, base_addr, dataset, exp_type)



    if mode == 'figure':
        df = pd.concat(df_list)
        return df
    elif mode == 'last_ten_epoch':
        print(df)
        if f_name:
            folder = f"./result/{f_name}/"
            if not os.path.exists(folder):
                os.system(f"mkdir -p {folder}")
            df.to_csv(f'{folder}{arch}_DivideMix_vs_proposed_table_{opt}.csv')
    else:
        raise NotImplementedError
