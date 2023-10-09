from transformers import pipeline

translation = pipeline("translation_de_to_en", model="Helsinki-NLP/opus-mt-de-en", tokenizer="Helsinki-NLP/opus-mt-de-en")

text = "Mein Name ist Anna. Ich komme aus Österreich und lebe seit drei Jahren in Deutschland. Ich bin 15 Jahre alt und habe zwei Geschwister: Meine Schwester heißt Klara und ist 13 Jahre alt, mein Bruder Michael ist 18 Jahre alt. Wir wohnen mit unseren Eltern in einem Haus in der Nähe von München. Meine Mutter ist Köchin, mein Vater arbeitet in einer Bank. Meine Familie hat verheerende Ereignisse durchgemacht, aber wir haben es geschafft, trotz allem zusammenzuhalten."
translated_text = translation(text, max_length=100)[0]['translation_text']
print(translated_text)