import pickle
import numpy as np
import os
import random
from common.util import cos_similarity
from better_rnnlm import BetterRnnlm
model_file = './aozoraBetterRnnlm_large.pkl'
vocab_file = './aozora_vocab.pkl'

if not os.path.exists(vocab_file) or not os.path.exists(model_file):
    print("Error: 必要なファイルが見つかりません。")
    exit()

with open(vocab_file, 'rb') as f:
    word_to_id1, id_to_word1 = pickle.load(f)

# モデルのセットアップ
vocab_size = len(word_to_id1)
model = BetterRnnlm(vocab_size, 650, 650)
model.load_params(model_file)
word_matrix1 = model.layers[0].W

# 既出管理用
used_words1 = set()
game_log = []

def word_arithmetic_logic(word1, word2, word3, word_to_id2, id_to_word2, word_matrix2, used_words2):
    for w in [word1, word2, word3]:
        if w not in word_to_id2: return None

    vecs = [word_matrix2[word_to_id2[w]] for w in [word1, word2, word3]]
    target_vec = vecs[0] - vecs[1] + vecs[2]

    similarity = np.dot(word_matrix2, target_vec) / (
                np.linalg.norm(word_matrix2, axis=1) * np.linalg.norm(target_vec) + 1e-8)
    sorted_indices = (-1 * similarity).argsort()

    for j in sorted_indices:
        res = id_to_word2[j]
        if res not in [word1, word2, word3] and res not in used_words2:
            return res
    return None


def get_next_word(query, word_to_id3, id_to_word3, word_matrix3, used_words3):
    if query not in word_to_id3:
        return random.choice(list(word_to_id3.keys()))

    # 1. まず純粋な類義語を探す
    query_vec = word_matrix3[word_to_id3[query]]
    similarity = np.dot(word_matrix3, query_vec) / (
                np.linalg.norm(word_matrix3, axis=1) * np.linalg.norm(query_vec) + 1e-8)
    sorted_indices = (-1 * similarity).argsort()

    # 自分自身を除いた最も近い単語を候補にする
    candidate_word = id_to_word3[sorted_indices[random.randint(1,5)]]

    # 2. 足し引き算を実行 (入力 - 類似語 + 太陽)
    # 固定で「太陽」を足す
    res = word_arithmetic_logic(query, candidate_word, "太陽", word_to_id3, id_to_word3, word_matrix3, used_words3)

    return res if res else candidate_word


def check_similarity(word1, word2, word_to_id4, word_matrix4):
    if word1 not in word_to_id4 or word2 not in word_to_id4: return 0
    v1, v2 = word_matrix4[word_to_id4[word1]], word_matrix4[word_to_id4[word2]]
    return cos_similarity(v1, v2)


# --- 3. メインループ ---
current_word = 'バナナ'
used_words1.add(current_word)

print("--- ロジカルバナナ！ ---")

for i in range(4):
    input_word = input(f"プレイヤー({i + 1}/10)：【{current_word}】といえば？ > ")

    score = check_similarity(current_word, input_word, word_to_id1, word_matrix1)
    used_words1.add(input_word)

    print(f"{'　　　　　　　　　　いいね！' if score >= 0.25 else '　　　　　　　　　　似てないよ～'} (類似度: {score:.3f})")

    # 生成
    next_word = get_next_word(input_word, word_to_id1, id_to_word1, word_matrix1, used_words1)
    used_words1.add(next_word)

    game_log.append({
        "turn": i + 1,
        "base": current_word,
        "input": input_word,
        "score": score,
        "ai": next_word
    })

    current_word = next_word
    print(f"　Ａ　Ｉ　({i + 1}/10)：【{input_word}】といえば？ > {current_word}")

# --- 4. ログ出力 ---
print("\n" + "=" * 60)
print(f"{'回':<3} | {'お題':<10} | {'あなたの回答':<10} | {'類似度':<6} | {'AIの連想'}")
print("-" * 60)
for l in game_log:
    print(f"{l['turn']:<3} | {l['base']:<10} | {l['input']:<10} | {l['score']:<8.3f} | {l['ai']}")
print("=" * 60)