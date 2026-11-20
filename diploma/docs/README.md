# Паспорт НИР

Из каталога `diploma` установите зависимости командой `make install-macos` или `make install-ubuntu`,
затем запустите `make docs`. Готовые файлы появятся в `docs/research-passport.pdf` и
`docs/research-passport.docx`. Команда `make pdf` собирает только PDF.

На macOS для установки нужны Homebrew и Make. На Ubuntu нужны Make и доступ к sudo.
Если Make ещё не установлен, выполните `sudo apt-get install make`.

## Оформление

За основу взяты разделы 3.2–3.3 [методических указаний Новотроицкого филиала МИСИС][guidelines].
Формат листа — А4. Поля слева и справа — 30 и 15 мм, сверху и снизу — по 20 мм.
Основной текст — 14 пт, по ширине, с отступом первой строки 12,5 мм и полуторным интервалом.
Заголовок — чёрный, полужирный, 16 пт, по центру. Номер на первой странице отсутствует.

В PDF используется Times New Roman, а при его отсутствии — метрический аналог Liberation Serif.
Для сборки DOCX Pandoc берёт стили Times New Roman и параметры страницы из `docs/reference.docx`.
Перед сдачей ВКР уточните требования выпускающей кафедры.

[guidelines]:
https://nf.misis.ru/download/gisen/%D0%98%D0%B7%D0%BC%D0%B0%D0%B9%D0%BB%D0%BE%D0%B2%D0%B0%2C_%D0%A2%D0%BE%D1%80%D1%88%D0%B8%D0%BD%D0%B0_%D0%BC%D0%B5%D1%82%D0%BE%D0%B4._%D1%83%D0%BA%D0%B0%D0%B7._%D0%BF%D0%BE_%D0%BE%D1%84%D0%BE%D1%80%D0%BC%D0%BB%D0%B5%D0%BD%D0%B8%D1%8E_%D0%92%D0%9A%D0%A0_2022.pdf
