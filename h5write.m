function h5write(data, filename)

% If you want to keep the axis order the same as
% in python, uncomment the following.
% Otherwise, axis is completely fliped in order
%data = permute(data, fliplr([1:ndims(A)]));

if isreal(data)
    hdf5write(filename, '/real', data);
else
    hdf5write(filename, 'imag', imag(data), 'WriteMode', 'overwrite');
    hdf5write(filename, 'real', real(data), 'WriteMode', 'append');
end