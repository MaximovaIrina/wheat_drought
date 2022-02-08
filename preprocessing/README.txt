1. python preprocessing/structuring.py
2. ''' EXTRACT TIR & RGB FROM IRSOFT TO STRUCT_PNG FOLDER '''
3. python preprocessing/calibration.py
4. 


torch.manual_seed(seed) - фиксиует веса


Разбиение по классификционным (0,1) данным 
кАЛИБРОВКА
сложности увантований
Рассматриваем все 3 сигмы(а не 2) так как в них еще содержится инфа для азделения (см гистограммы квантованные)

вторая производная: так как читаем во флотах, то точного х где производная равна 0 мы не знаем, но можем найти если x[i]<0 x[i+1]>0
Добавить код трешолдига

Сужен диапазон 18-23 (в основном 19-20 растения, но изза колебаний и помех взяты шире) + картинка Никита

21/6 = train/test

    # TODO: 
    # inference time + weight feach
    # feature visualize
    # Mertic dependences from group/hidden_n for diff HISTQUAN with fix number bins

    # space_mode | ndvi_mode | task_mode | bins_path | hidden_n | features_group