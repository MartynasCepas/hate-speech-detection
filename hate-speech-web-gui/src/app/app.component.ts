import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'hate-speech-web-gui';

  textToCheck: string = '';
  result: string | null = null;
  isHateSpeech: boolean = false;

  constructor(private http: HttpClient) {}

  onSubmit() {
    this.http.post<any>('/predict', { text: this.textToCheck }).subscribe({
      next: (response) => {
        this.isHateSpeech = response.result === 'Hate speech';
        this.result = this.isHateSpeech ? 'Neapykantos kalba aptikta' : 'Neapykantos kalba neaptikta';
      },
      error: (error) => {
        console.error('Error:', error);
        this.result = 'Klaida nustatant neapykantos kalbą';
        this.isHateSpeech = true;
      },
      complete: () => console.log('Request completed') // Optional
    });
  }
}
