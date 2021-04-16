


def get_prediction(sentence):
    tokens = encode_sentence(sentence)

    input_ids = get_ids(tokens)
    input_mask = get_mask(tokens)
    segment_ids = get_segments(tokens)

    inputs = tf.stack(
        [tf.cast(input_ids, dtype=tf.int32),
         tf.cast(input_mask, dtype=tf.int32),
         tf.cast(segment_ids, dtype=tf.int32)],
         axis=0)
    inputs = tf.expand_dims(inputs, 0) # simula un lote

    output = Dcnn(inputs, training=False)

    sentiment = math.floor(output*2)

    if sentiment == 0:
        print("Salida del modelo: {}\nSentimiento predicho: Negativo.".format(
            output))
    elif sentiment == 1:
        print("Salida del modelo: {}\nSentimiento predicho: Positivo.".format(
            output))