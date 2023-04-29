
def sorted_dict_by_value(dictionary):
    return {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1])}
