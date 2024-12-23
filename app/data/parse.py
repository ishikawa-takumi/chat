import re
import pdfplumber
import json
import os
import sys

# PDFファイルパス
pdf_file_path = sys.argv[1]

# ファイル存在チェック
if not os.path.exists(pdf_file_path):
    raise FileNotFoundError(f"指定されたファイルが見つかりません: {pdf_file_path}")

# パースしたデータを格納するリスト
parsed_data = []

# 正規表現パターン
user_pattern = r"(\S+ \S+)\(ID:(\d+)\)"
timestamp_pattern = r"(\d{4}年\d{2}月\d{2}日 \d{2}:\d{2})"
message_block_pattern = r"(.*?)\n(?=\d{4}年|\[info\]|\[qt\]|\[rp\]|$)"
info_pattern = r"\[info\](.*?)\[/info\]"
qt_pattern = r"\[qt\](.*?)\[/qt\]"
rp_pattern = r"\[rp aid=(\d+) to=(\d+)](.*?)$"

def clean_username(username):
    """名前の重複を防ぐため、文字列をクリーンアップ"""
    return re.sub(r"(.+?)\1", r"\1", username)

def parse_message_block(block):
    """各メッセージブロックを解析してデータ構造を生成"""
    message = re.search(message_block_pattern, block, re.DOTALL)
    info_matches = re.findall(info_pattern, block, re.DOTALL)
    qt_matches = re.findall(qt_pattern, block, re.DOTALL)
    rp_matches = re.findall(rp_pattern, block, re.DOTALL)

    return {
        "message": message.group(1).strip() if message else None,
        "info": info_matches,
        "quotes": qt_matches,
        "replies": [
            {"reply_to_id": rp[1], "reply_content": rp[2]} for rp in rp_matches
        ]
    }

def parse_page(page_text):
    # convert Shift-JIS to UTF-8
    # page_text = page_text.encode('cp932', 'ignore').decode('utf-8', 'ignore')

    """1ページ分のテキストを解析して構造化データに変換"""
    results = []
    # 各発言者ごとに分割
    user_blocks = re.split(user_pattern, page_text)
    
    for i in range(1, len(user_blocks), 3):
        username = clean_username(user_blocks[i].strip())
        user_id = user_blocks[i+1].strip()
        content_block = user_blocks[i+2]

        # タイムスタンプを抽出
        timestamp_match = re.search(timestamp_pattern, content_block)
        timestamp = timestamp_match.group(1) if timestamp_match else None

        # メッセージブロックを解析
        parsed_message = parse_message_block(content_block)

        # データ構造化
        result = {
            "username": username,
            "user_id": user_id,
            "timestamp": timestamp,
            "message": parsed_message["message"],
            "info": parsed_message["info"],
            "quotes": parsed_message["quotes"],
            "replies": parsed_message["replies"]
        }
        results.append(result)
    return results

def format_entry(entry):
    """
    JSONエントリをテキスト形式に整形
    """
    formatted_entry = [
        f"ユーザー名: {entry.get('username', 'なし')}",
        f"ユーザーID: {entry.get('user_id', 'なし')}",
        f"タイムスタンプ: {entry.get('timestamp', 'なし')}",
        f"メッセージ: {entry.get('message', 'なし')}",
        f"補足情報: {' '.join(entry.get('info', [])) if entry.get('info') else 'なし'}",
        f"引用: {' '.join(entry.get('quotes', [])) if entry.get('quotes') else 'なし'}",
        f"返信: {' '.join(entry.get('replies', [])) if entry.get('replies') else 'なし'}",
    ]
    return "\n".join(formatted_entry) + "\n***************************\n"


# PDFを開いて解析
with pdfplumber.open(pdf_file_path) as pdf:
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:  # テキストが存在する場合にのみ解析
            page_data = parse_page(page_text)
            parsed_data.extend(page_data)


# 結果をJSONに保存
# output_path = "parsed_chat.json"
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(parsed_data, f, ensure_ascii=False, indent=4)
# print(f"解析が完了しました。結果は {output_path} に保存されました。")

documents = [format_entry(entry) for entry in parsed_data]

# Open the file with UTF-8 encoding
with open("output.txt", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc)