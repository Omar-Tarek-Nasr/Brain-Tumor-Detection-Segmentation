// معاينة الصورة قبل الرفع
document.getElementById('imageUpload').addEventListener('change', function(e) {
    const preview = document.getElementById('imagePreview');
    if (this.files && this.files[0]) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.classList.remove('d-none');
        }
        
        reader.readAsDataURL(this.files[0]);
    }
});