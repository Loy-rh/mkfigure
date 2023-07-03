import numpy as np
import pandas as pd
import os
import glob


def mk_csv(df, method, f_name, arch, ylabel):
    assert f_name is not None
    folder = "./result/{}/".format(f_name)
    if not os.path.exists(folder):
        # os.system(f"mkdir -p {folder}")
        os.system("mkdir -p {}".format(folder))
    df.to_csv('{}{}_{}_{}_table.csv'.format(folder, arch, method, ylabel))
    print("save to {}{}_{}_{}_table.csv".format(folder, arch, method, ylabel))


def final_df(Ylabel, ylabel, dataset, arch, hue_order, mode='figure', opt='average',
             f_name=None, root=None, read_dir=None, exp_type_order=None):
    df_list = []

    # change the order of the graphs by changing this exp_type order
    exp_type_order2 = []
    exp_dict = {}
    for type in exp_type_order:
        if type == "0.2_sym":
            exp_type_order2.append("sym_20")
            exp_dict.update({"0.2_sym": "Symmetric 20%"})
        elif type == "0.4_sym":
            exp_type_order2.append("sym_40")
            exp_dict.update({"0.4_sym": "Symmetric 40%"})
        elif type == "0.5_sym":
            exp_type_order2.append("sym_50")
            exp_dict.update({"0.5_sym": "Symmetric 50%"})
        elif type == "0.8_sym":
            exp_type_order2.append("sym_80")
            exp_dict.update({"0.8_sym": "Symmetric 80%"})
        elif type == "0.9_sym":
            exp_type_order2.append("sym_90")
            exp_dict.update({"0.9_sym": "Symmetric 90%"})
        elif type == "0.2_asym":
            exp_type_order2.append("asym_20")
            exp_dict.update({"0.2_asym": "Asymmetric 20%"})
        elif type == "0.4_asym":
            exp_type_order2.append("asym_40")
            exp_dict.update({"0.4_asym": "Asymmetric 40%"})
    # exp_dict = {"0.2_sym": "Symmetric 20%", "0.5_sym": "Symmetric 50%",
    #             "0.8_sym": "Symmetric 80%", "0.9_sym": "Symmetric 90%",
    #             "0.4_asym": "Asymmetric 40%"}

    if mode == 'figure':
        def make_df_plus(method, base_addr, exp_type):
            if ylabel == 'Accuracy':
                path = glob.glob("{}/*acc.txt".format(base_addr))[0]
                assert os.path.isfile(path)
                df1 = pd.read_csv(path, delim_whitespace=True)
            elif ylabel == 'Number_of_labeled_samples':
                path = glob.glob("{}/*states1.txt".format(base_addr))[0]
                assert os.path.isfile(path)
                df1 = pd.read_csv(path, delim_whitespace=True)
                df1[ylabel] = [i/500 for i in df1[ylabel]]
            # elif ylabel == "AUC":
            #     path = glob.glob("{}/*states1.txt".format(base_addr))[0]
            #     assert os.path.isfile(path)
            #     df1 = pd.read_csv(path, delim_whitespace=True)
            else:
                path = glob.glob("{}/*states1.txt".format(base_addr))[0]
                assert os.path.isfile(path)
                df1 = pd.read_csv(path, delim_whitespace=True)

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
                len(exp_dict) * len(hue_order), dtype=np.float
            ).reshape(2, len(exp_dict)),
            index=["Best", "Last"], columns=list(map(lambda key: exp_dict[key], exp_type_order))
        )

        def last_ten_epoch(method, base_addr, exp_type):
            total = 0
            if ylabel == 'Accuracy':
                assert len(glob.glob("{}/*acc.txt".format(base_addr))) != 0
                path = glob.glob("{}/*acc.txt".format(base_addr))[0]
                assert os.path.isfile(path)
                df1 = pd.read_csv(path, delim_whitespace=True)
            elif ylabel == "AUC":
                assert len(glob.glob("{}/*acc.txt".format(base_addr))) != 0
                path = glob.glob("{}/*states1.txt".format(base_addr))[0]
                assert os.path.isfile(path)
                df1 = pd.read_csv(path, delim_whitespace=True)
            total += df1[ylabel][-10:]
            from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
            # general rounding
            avg = Decimal(np.mean(total)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            df.at["Last", exp_dict[exp_type]] = avg
            df.at["Best", exp_dict[exp_type]] = np.max(df1[ylabel])
        process = last_ten_epoch
    else:
        raise NotImplementedError

    # for exp_type, exp_type2 in zip(exp_type_order, exp_type_order2):
    #     method = 'DivideMix'
    #     if method in hue_order:
    #         base_addr = "{}{}/log/{}_{}_{}_{}/*".format(
    #             root, read_dir[method], dataset, method, arch, exp_type2)
    #         process(method, base_addr, exp_type)
    #         if mode == 'last_ten_epoch':
    #     method = 'Proposed'
    #     if method in hue_order:
    #         base_addr = "{}{}/log/{}_UPL_{}_{}/*".format(
    #             root, read_dir[method], dataset, arch, exp_type2)
    #         process(method, base_addr, exp_type)
    #         if mode == 'last_ten_epoch':
    #             mk_csv(df, method, f_name, arch)
    for method in hue_order:
        if method == 'DivideMix':
            for exp_type, exp_type2 in zip(exp_type_order, exp_type_order2):
                base_addr = "{}{}/log/{}_{}_{}_{}/*".format(
                    root, read_dir[method], dataset, method, arch, exp_type2)
                process(method, base_addr, exp_type)
            if mode == 'last_ten_epoch':
                mk_csv(df, method, f_name, arch, ylabel)
        elif method == 'Proposed':
            for exp_type, exp_type2 in zip(exp_type_order, exp_type_order2):
                base_addr = "{}{}/log/{}_UPL_{}_{}/*".format(
                    root, read_dir[method], dataset, arch, exp_type2)
                print(base_addr, '\n')
                process(method, base_addr, exp_type)
            if mode == 'last_ten_epoch':
                mk_csv(df, method, f_name, arch, ylabel)
        elif method == 'CRAS':
            for exp_type, exp_type2 in zip(exp_type_order, exp_type_order2):
                base_addr = "{}{}/log/{}_CRAS_{}_{}/*".format(
                    root, read_dir[method], dataset, arch, exp_type2)
                process(method, base_addr, exp_type)
            if mode == 'last_ten_epoch':
                mk_csv(df, method, f_name, arch, ylabel)
        elif method == 'UPL' or 'UPLplus':
            for exp_type, exp_type2 in zip(exp_type_order, exp_type_order2):
                base_addr = "{}{}/log/{}_{}_{}_{}/*".format(
                    root, read_dir[method], dataset, method, arch, exp_type2)
                process(method, base_addr, exp_type)
            if mode == 'last_ten_epoch':
                mk_csv(df, method, f_name, arch, ylabel)
        else:
            for exp_type, exp_type2 in zip(exp_type_order, exp_type_order2):
                base_addr = "{}{}/log/{}_UPL_{}_{}/*".format(
                    root, read_dir[method], dataset, arch, exp_type2)
                print(base_addr, '\n')
                process(method, base_addr, exp_type)
            if mode == 'last_ten_epoch':
                mk_csv(df, method, f_name, arch, ylabel)

    if mode == 'figure':
        df = pd.concat(df_list)
        return df
    elif mode == 'last_ten_epoch':
        pass
    #     print(df)
    #     if f_name:
    #         folder = f"./result/{f_name}/"
    #         if not os.path.exists(folder):
    #             os.system(f"mkdir -p {folder}")
    #         df.to_csv(f'{folder}{arch}_DivideMix_vs_proposed_table_{opt}.csv')
    else:
        raise NotImplementedError
