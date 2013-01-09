function h5write(data, filename)

if isreal(data)
    hdf5write(filename, '/real', data.');
else
    hdf5write(filename, 'imag', imag(data).','WriteMode','overwrite');
    hdf5write(filename, 'real', real(data).','WriteMode','append');
    
end