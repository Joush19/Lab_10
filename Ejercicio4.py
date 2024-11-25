import pandas as pd
from functools import reduce

data = {
    'Student Name': ['Juan', 'Maria', 'Pedro', 'Ana'],
    'Student Age': [20, 22, 21, 23],
    'No. of Lab completed': [5, 7, 6, 8],
    'Average score': [85.5, 90.0, 88.0, 92.5]
}

df = pd.DataFrame(data)

df.to_excel('students_data.xlsx', index=False)

print("Archivo Excel generado con éxito.")

def load_data(file_name):
    return pd.read_excel(file_name)

def display_data(df):
    return df

def filter_students_by_age(df, age):
    return df[df['Student Age'] > age]

def average_score(df):
    return df['Average score'].mean()

def process_data(file_name):

    df = load_data(file_name)

    students_info = list(map(lambda row: f"Nombre: {row['Student Name']}, Edad: {row['Student Age']}, "
                                        f"Laboratorios completados: {row['No. of Lab completed']}, "
                                        f"Promedio: {row['Average score']}", df.to_dict(orient='records')))

    print("Información de los estudiantes:")
    for student in students_info:
        print(student)

    filtered_students = filter_students_by_age(df, 21)
    print("\nEstudiantes mayores de 21 años:")
    print(display_data(filtered_students))

    avg_score = reduce(lambda x, y: x + y, df['Average score']) / len(df)
    print(f"\nPromedio de las notas de los estudiantes: {avg_score:.2f}")

process_data('students_data.xlsx')
