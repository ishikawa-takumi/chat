FROM python:3.10-slim

# 作業ディレクトリ
WORKDIR /app

# 必要なライブラリをインストール
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . /app/

# ポートを開放
EXPOSE 8000

# 起動コマンド
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
