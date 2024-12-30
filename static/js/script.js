document.getElementById('login-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    fetch('/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            window.location.href = '/add-pegawai'; // Redirect to add pegawai page
        } else {
            alert('Login failed: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred during login.');
    });
});

// Submit form to add pegawai
document.getElementById('add-pegawai-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const formData = new FormData();
    formData.append('nama', document.getElementById('nama').value);
    formData.append('nip', document.getElementById('nip').value);
    formData.append('alamat', document.getElementById('alamat').value);
    formData.append('nomor_telepon', document.getElementById('nomor_telepon').value);

    fetch('/add_pegawai', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Pegawai added successfully');
            window.location.href = '/get-pegawai'; // Redirect to pegawai list page
        } else {
            alert('Failed to add pegawai: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while adding pegawai.');
    });
});

