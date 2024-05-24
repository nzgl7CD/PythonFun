my_dictionary={}

def add_my_family(role, name):
    if name.upper() in my_dictionary:
        my_dictionary[role]=[my_dictionary[role], name]

    else:
        my_dictionary[role]=name

