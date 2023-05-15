from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

opinia1 = "Booked for a surprise for my boyfriend’s birthday and Everything was perfect, from the decor to the amazing staff who go above and beyond. The rooms are unreal and lovely! Spotless and LOVED the extras in the room! My boyfriend was so happy after the trip. Thank you Gotham we will be back x"
opinia2 = " cleaning staff walked in on us at fucing 7am and again at 9am. some confusion about whether the room was occupied. was assured they would not disturb us again before checkout at 11am. At 10:40 housekeeping knocked to our fucking door again. Wouldn’t stay there again."

analyzer = SentimentIntensityAnalyzer()

scores1 = analyzer.polarity_scores(opinia1)
print(f"Opinia 1: {scores1}")
print(type(scores1))

scores2 = analyzer.polarity_scores(opinia2)
print(f"Opinia 2: {scores2}")

import text2emotion as te

emocje1 = te.get_emotion(opinia1)
print(f"Opinia 1: {emocje1}")

emocje2 = te.get_emotion(opinia2)
print(f"Opinia 2: {emocje2}")
