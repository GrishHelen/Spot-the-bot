# Spot-the-bot

Курсовая работа, 2 курс ПМИ, НИУ ВШЭ

Полный текст курсовой работы с обзором литературы, описанием методов и полученными результатами доступен в файле `Coursework_text.pdf`.

### Введение

Обработка естественного языка (NLP) - направление на стыке лингвистики, математики и машинного обучения, которое исследует и обрабатывает на семантическом уровне тексты, написанные людьми. В рамках проекта "Spot the bot"мы исследуем семантические пространства языков разных групп и сравниваем их между собой.

Кроме того, в настоящее время приближение поведения разговорного ИИ к человеческому – актуальная задача, и со стороны пользователей есть запрос на более человечное поведение чат-ботов. Именно поэтому важно научить бота распознавать юмор и придумывать новый смешной ответ.

Некоторые современные исследования направлены на генерацию определенных типов юмористических текстов на основе объемных размеченных данных или используют для этой цели преимущественно методы обучения с учителем. Такие модели нестабильны, слабо обобщаются на разные виды юмора и способны только воспроизводить известные сценарии шуток. Однако, насколько нам известно, пока не существует достаточно полных исследований семантических пространств юмористических текстов естественных языков. Нахождение структуры семантического пространства юмористических текстов может помочь в генерации новых шуток.

#### Гипотеза

Предполагается, что шутки расположены на границах дыр в пространстве векторов, соответствующих n-граммам, в то время как нейтральные тексты находятся дальше от них. Для того, чтобы проверить эту гипотезу, требуется собрать информативный корпус текстов, который дает исчерпывающее представление о семантической структуре языка. Также необходимо научиться восстанавливать тексты по выделенным персистентным гомологиям, причем делать это так, чтобы фразы оставались юмористическими и были похожи на тексты естественного языка

### Часть 1. Числа Бетти.

Наша основная задача - исследовать топологическую структуру пространства языка, поэтому мы будем рассматривать его различные характеристики. Начнем с чисел Бетти, которые соответствуют количеству p-мерных ‘дырок’ в векторных пространства. Для вычисления чисел Бетти на практике используются boundary матрицы и вычисляют их ранги.

### Часть 2. Выделение персистентных гомологий

Неформально говоря, персистентная гомология содержит информацию о ‘дырах’ пространства, включая время их появления и исчезновения, размерность и расположение. Также персистентная гомология отслеживает ϵ, при которых появляются и исчезают дырки пространства комплекса. 

Задача построения гомологий рассматривается на заданном множестве точек в p-мерном пространстве с заданным на нем симплициальным комплексом. В этой работе был рассмотрен комплекс Вьеториса-Рипса. Мы хотим научиться окантовывать ‘дыры’ в пространстве векторов n-грамм, полученных из текстов естественных языков, и идентифицировать их положение в пространстве и относительно других векторов.

### Направления дальнейшей работы

Мы планируем продолжить анализ пространства слов русского языка, провести исследования для других языков и сравнить результаты, полученные на пространствах слов различных естественных языков. Такоже на основе результатов, полученных из анализа топологической структуры различных текстов, планируется генерировать шутки (фразы, расположенные около границ дыр пространства n-грамм), учитывая выявленные характерные свойства юмористических текстов.
