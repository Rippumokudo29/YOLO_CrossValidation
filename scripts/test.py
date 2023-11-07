if __name__ == '__main__':
    
    label_path = 'datasets/20231001_4chamber/labels/train/Male_BMIu25_normal (30)_AS000027_36.txt'
    
    list_true_data = []
    with open(label_path) as f:
        for line in f:
            list_true_data.append(line.split())
    f.close()
    print(list_true_data)
    
    data = None
    
    if not data:
        print('ok')