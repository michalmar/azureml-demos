def generate_big_random_bin_file(filename,size):
    """
    generate big binary file with the specified size in bytes
    :param filename: the filename
    :param size: the size in bytes
    :return:void
    """
    import os 
    with open('%s'%filename, 'wb') as fout:
        fout.write(os.urandom(size)) #1

    print (f'big random binary file with size {size} generated ok')
    pass

if __name__ == '__main__':
    gb_size = 20
    generate_big_random_bin_file(f"/tmp/mma/file_{gb_size}gb.dat",gb_size*1024*(1024*1024))