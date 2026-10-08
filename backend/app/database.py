import psycopg
DATABASE_URL="postgresql://admin:admin123@database:5432/company_ai"
def get_connection():
    return psycopg.connect(DATABASE_URL)