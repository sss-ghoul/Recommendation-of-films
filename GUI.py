from movie import *
import PySimpleGUI as sg

# Определяем слои ГУИ
layout = [
    [sg.Text("Введите свой запрос по английски:"), sg.InputText(key='-INPUT-')],
    [sg.Text("Количество результатов:"), sg.Combo(values=[5, 10, 15], default_value=5, key='-NUM_RESULTS-')],
    [sg.Text("Методы"), sg.Combo(values=['Косинусное подобие','Расстояние между векторами','Метод ближайших соседей'], default_value='Косинусное подобие', key='-FUNC-')],
    [sg.Button("Найти"), sg.Button("Очистить")],
    [sg.Text("Результаты поиска:")],
    [sg.Output(size=(100, 40), key='-OUTPUT-')]
]

# Создаем окно
window = sg.Window("Система рекомендаций фильмов", layout)

# Цикл событий для обработки событий и взаимодействия с графическим интерфейсом.
while True:
    event, values = window.read()

    # Если окно закрыто или нажата кнопка «Очистить», выйдите из цикла
    if event == sg.WINDOW_CLOSED :
        break


    if event == "Очистить":
        window['-OUTPUT-'].update("")
        # продолжаем

    # прописываем хэндлеры кнопок
    if event == "Найти":
        
        nos=values['-NUM_RESULTS-']
        
        algo=values['-FUNC-']
        
        query = values['-INPUT-']
        if algo == 'Косинусное подобие':
            Output_Title,Output_Summary=similarity_search_Cos_Sim(query,nos)
        if algo == 'Расстояние между векторами':
            Output_Title,Output_Summary=similarity_search_pinecone(query,nos)
        if algo == 'Метод ближайших соседей':
            Output_Title,Output_Summary=similarity_search_ANN(query,nos)


        # Печатаем результаты в поле выхода
        for _ in range(len(Output_Title)):
            print("Title: {}".format(Output_Title[_]))
            print("-" * 50)  # ставим разделитель
            print("Summary: {}".format(Output_Summary[_]))
            print()



# закрываем окно ГУИ
window.close()
